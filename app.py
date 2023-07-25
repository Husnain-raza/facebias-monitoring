from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import roc_curve
import json

import matplotlib.pyplot as plt



import os
import numpy as np
import pandas as pd
import tensorflow as tf


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/'


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        model = request.files['model']
        x_test = request.files['x_test']
        y_test = request.files['y_test']

        if not all(
                [model.filename.endswith('.h5'), x_test.filename.endswith('.npy'), y_test.filename.endswith('.npy')]):
            return "All files must be in the correct format (.h5 for model, .npy for data)."

        model_filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(model.filename))
        x_test_filename = os.path.join(app.config['UPLOAD_FOLDER'],  secure_filename(x_test.filename))
        y_test_filename = os.path.join(app.config['UPLOAD_FOLDER'],  secure_filename(y_test.filename))

        model.save(model_filename)
        x_test.save(x_test_filename)
        y_test.save(y_test_filename)

        return redirect(
            url_for('loading', model_path=model_filename, x_test_path=x_test_filename, y_test_path=y_test_filename))

    return render_template('upload.html')





@app.route('/monitor', methods=['GET'])
def monitor():
    model_path = request.args.get('model_path')
    x_test_path = request.args.get('x_test_path')
    y_test_path = request.args.get('y_test_path')

    # Load model and data
    model = tf.keras.models.load_model(model_path)
    x_test = np.load(x_test_path)
    y_test = np.load(y_test_path)[:600]

    # Make predictions
    y_pred = model.predict(x_test)
    y_pred_class = (y_pred > 0.5).astype(int)

    # Compute overall evaluation metrics
    accuracy = accuracy_score(y_test, y_pred_class)
    precision = precision_score(y_test, y_pred_class)
    recall = recall_score(y_test, y_pred_class)
    f1 = f1_score(y_test, y_pred_class)
    auc_roc = roc_auc_score(y_test, y_pred)

    # Create BinaryLabelDataset for the original test data
    bld_orig = BinaryLabelDataset(favorable_label=1,
                                  unfavorable_label=0,
                                  df=pd.DataFrame(data=np.hstack([y_test.reshape(-1,1), y_test.reshape(-1,1)]),
                                                  columns=['predicted_label', 'true_label']),
                                  label_names=['predicted_label'],
                                  protected_attribute_names=['true_label'],
                                  unprivileged_protected_attributes=[0])

    # Create BinaryLabelDataset for the classified test data
    bld_classif = BinaryLabelDataset(favorable_label=1,
                                     unfavorable_label=0,
                                     df=pd.DataFrame(data=np.hstack([y_pred_class.reshape(-1,1), y_test.reshape(-1,1)]),
                                                     columns=['predicted_label', 'true_label']),
                                     label_names=['predicted_label'],
                                     protected_attribute_names=['true_label'],
                                     unprivileged_protected_attributes=[0])

    # Compute bias metrics
    metric = ClassificationMetric(bld_orig, bld_classif,
                                  unprivileged_groups=[{'true_label': 0}],
                                  privileged_groups=[{'true_label': 1}])
    disparate_impact = metric.disparate_impact()
    equal_opportunity_difference = metric.equal_opportunity_difference()

    # Generating Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_class)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title('Confusion Matrix')
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url1 = base64.b64encode(img.getvalue()).decode()

    # Generating ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url2 = base64.b64encode(img.getvalue()).decode()

    # Render the dashboard
    return render_template('dashboard.html',
                           accuracy=accuracy, precision=precision, recall=recall,
                           f1=f1, auc_roc=auc_roc, disparate_impact=disparate_impact,
                           equal_opportunity_difference=equal_opportunity_difference,
                           plot1=plot_url1, plot2=plot_url2)


if __name__ == "__main__":
    app.run(debug=True)
