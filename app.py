from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
from sklearn.metrics import accuracy_score
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import textwrap

app = Flask(__name__)

CSV_FILE = "stock_data.csv"
df = pd.read_csv(CSV_FILE)

FEATURE_NAMES = ["price_change", "volume", "momentum", "trend_num"]
TARGET = "expected"

if "trend" in df.columns:
    df["trend_num"] = df["trend"].map({"up": 1, "down": -1, "neutral": 0})

X = df[FEATURE_NAMES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.8, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

rules_list = []
for idx, estimator in enumerate(clf.estimators_):
    tree_rules = export_text(estimator, feature_names=list(X.columns), decimals=1)
    rules_list.append(f"--- Tree {idx+1} ---\n{tree_rules}")

rules = "\n\n".join(rules_list)

def compute_node_depths(tree):
    """
    Compute depth for each node in a decision tree.
    Returns a dict {node_id: depth}.
    """
    node_depth = {}
    stack = [(0, 0)]  

    while stack:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        if left_child != -1:
            stack.append((left_child, depth + 1))
        if right_child != -1:
            stack.append((right_child, depth + 1))

    return node_depth

def most_common_paths(clf, X_np, feature_names, max_paths=10):
    path_rows = defaultdict(set)

    for estimator in clf.estimators_:
        tree = estimator.tree_
        node_depths = compute_node_depths(tree)

        node_indicator = estimator.decision_path(X_np)
        leaf_ids = estimator.apply(X_np)

        for sample_id in range(X_np.shape[0]):
            node_index = node_indicator.indices[
                node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
            ]

            sorted_nodes = sorted(node_index, key=lambda x: node_depths[x])

            conditions = []
            for node_id in sorted_nodes:
                if tree.feature[node_id] != -2: 
                    feature = feature_names[tree.feature[node_id]]
                    threshold = tree.threshold[node_id]
                    if X_np[sample_id, tree.feature[node_id]] <= threshold:
                        conditions.append(f"{feature} <= {threshold:.1f}")
                    else:
                        conditions.append(f"{feature} > {threshold:.1f}")

            if len(conditions) < 2:
                continue

            leaf_id = leaf_ids[sample_id]
            values = tree.value[leaf_id][0]
            predicted_class = clf.classes_[values.argmax()]
            path_key = (" → ".join(conditions), predicted_class)
            path_rows[path_key].add(sample_id)

    top_paths = []
    for (path, pred_class), rows in path_rows.items():
        percent = len(rows) / X_np.shape[0] * 100
        top_paths.append((path, pred_class, len(rows), round(percent, 1)))

    top_paths.sort(key=lambda x: x[2], reverse=True)
    return top_paths[:max_paths]

@app.route("/")
def summary():
    os.makedirs("static", exist_ok=True)

    explainer = shap.Explainer(clf, X_train)
    shap_values = explainer(X_test)

    class_names = list(clf.classes_)
    beeswarm_paths = {}
    bar_paths = {}

    for class_idx, class_label in enumerate(class_names):

        plt.figure()
        shap.plots.beeswarm(shap_values[:, :, class_idx], show=False)
        beeswarm_file = f"shap_beeswarm_{class_label}.png"
        plt.savefig(os.path.join("static", beeswarm_file), dpi=150, bbox_inches="tight")
        plt.close()
        beeswarm_paths[class_label] = beeswarm_file

        plt.figure()
        shap.plots.bar(shap_values[:, :, class_idx], show=False)
        bar_file = f"shap_bar_{class_label}.png"
        plt.savefig(os.path.join("static", bar_file), dpi=150, bbox_inches="tight")
        plt.close()
        bar_paths[class_label] = bar_file

    common_paths = most_common_paths(clf, X_test.values, X_test.columns, max_paths=10)

    return render_template(
        "summary.html",
        accuracy=round(accuracy * 100, 2),
        rules=rules,
        common_paths=common_paths,
        beeswarm_paths=beeswarm_paths,
        bar_paths=bar_paths
    )

@app.route("/custom", methods=["GET", "POST"])
def custom():
    result = None
    if request.method == "POST":
        user_code = request.form["code"]

        indented_user_code = textwrap.indent(user_code, "    ")

        func_code = f"""
def stock_decision(stock):
    price_change = float(stock.get('price_change', 0))
    volume = float(stock.get('volume', 0))
    momentum = float(stock.get('momentum', 0))

    trend_num = stock.get('trend_num')
    if trend_num is None:
        t = stock.get('trend', 'neutral')
        trend_num = 1 if t == 'up' else -1 if t == 'down' else 0

{indented_user_code}
"""
        local_env = {}
        try:
            exec(func_code, local_env)
            stock_decision = local_env["stock_decision"]

            preds = []
            for _, row in df.iterrows():
                pred = stock_decision(row)
                preds.append(pred)

            acc = accuracy_score(df[TARGET], preds)
            result = f"Custom rule accuracy: {acc*100:.2f}%"
        except Exception as e:
            result = f"Error in your code: {e}"

    return render_template("custom.html", result=result)

@app.route("/test")
def test():
    return redirect("https://blockly-demo.appspot.com/static/demos/code/index.html")

if __name__ == "__main__":
    app.run(debug=True)
