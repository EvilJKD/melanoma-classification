import os
import numpy as np
import pandas as pd
from scipy import interpolate
import pickle as pck
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, ConfusionMatrixDisplay


# Plots
def loss_v_iteration_curve(epochs, training_loss_list, test_loss_list, title):
    epoch_count = range(1, epochs + 1)


    # Visualize loss history
    plt.plot(epoch_count, training_loss_list, 'r', label="Training Loss")
    plt.plot(epoch_count, test_loss_list, 'b', label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Mean of BC Loss')
    plt.legend(loc='best')
    plt.show()
    plt.close()


def auc_v_iterations_curve(epochs, training_auc_list, test_auc_list, title):
    epoch_count = range(1, epochs + 1)

    plt.plot(epoch_count, training_auc_list, 'r-')
    plt.plot(epoch_count, test_auc_list, 'b-')
    plt.legend(['Training AUC', 'Test AUC'])
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.savefig(title)
    plt.close()


def auc_curve_plot(y_test, y_pred, title):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    roc_auc = auc(fpr, tpr)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(roc_auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(title)
    plt.close()


def precision_recall_curve_plot(y_test, y_pred, title):
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

    plt.figure(2)
    plt.plot(recall, precision, color='purple')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PRE vs REC')
    plt.legend(loc='best')
    plt.savefig(title)
    plt.close()

def analize_mean_std(name, training, validation):


    tr_means = []
    tr_stds = []

    val_means = []
    val_stds = []

    for k in training[name].keys():
        tr_means.append(np.mean(training[name][k]))
        tr_stds.append(np.std(training[name][k]))

    for k in validation[f'val_{name}'].keys():
        val_means.append(np.mean(validation[f'val_{name}'][k]))
        val_stds.append(np.std(validation[f'val_{name}'][k]))



    with open(f'Results_{name}.csv', 'a+') as file:

        file.write(f"EPOCH,TrMean,TrSTD,ValMean,ValSTD\n")

        epoch = 1

        tr = zip(tr_means, tr_stds)
        val = zip(val_means, val_stds)
        for tr, val in zip(tr, val):
            file.write(
                f"{epoch},{tr[0]},{tr[1]},{val[0]},{val[1]}\n")
            epoch +=1

    return tr_means, val_means




def load_data_and_plot(history_dir, predictions_dir, epochs, model_name):
    metrics = {
        'loss': {},
        'val_loss': {},
        'auc': {},
        'val_auc': {},
        'acc': {},
        'val_acc': {},
        'pre': {},
        'val_pre': {},
        'rec': {},
        'val_rec': {}
    }

    files = os.listdir(history_dir)

    for file in files:
        if os.path.isfile(os.path.join(history_dir, file)):
            history = np.load(os.path.join(history_dir, file), allow_pickle='TRUE').item()

            for epoch in range(epochs):


                if epoch in metrics['loss'].keys():
                    metrics['loss'][epoch].append(history['loss'][epoch])
                else:
                    metrics['loss'][epoch] = [history['loss'][epoch]]

                if epoch in metrics['val_loss'].keys():
                    metrics['val_loss'][epoch].append(history['val_loss'][epoch])
                else:
                    metrics['val_loss'][epoch] = [history['val_loss'][epoch]]


                if epoch in metrics['auc'].keys():
                    metrics['auc'][epoch].append(history['auc'][epoch])
                else:
                    metrics['auc'][epoch] = [history['auc'][epoch]]

                if epoch in metrics['val_auc'].keys():
                    metrics['val_auc'][epoch].append(history['val_auc'][epoch])
                else:
                    metrics['val_auc'][epoch] = [history['val_auc'][epoch]]


                if epoch in metrics['acc'].keys():
                    metrics['acc'][epoch].append(history['accuracy'][epoch])
                else:
                    metrics['acc'][epoch] = [history['accuracy'][epoch]]

                if epoch in metrics['val_acc'].keys():
                    metrics['val_acc'][epoch].append(history['val_accuracy'][epoch])
                else:
                    metrics['val_acc'][epoch] = [history['val_accuracy'][epoch]]

                if epoch in metrics['pre'].keys():
                    metrics['pre'][epoch].append(history['precision'][epoch])
                else:
                    metrics['pre'][epoch] = [history['precision'][epoch]]

                if epoch in metrics['val_pre'].keys():
                    metrics['val_pre'][epoch].append(history['val_precision'][epoch])
                else:
                    metrics['val_pre'][epoch] = [history['val_precision'][epoch]]

                if epoch in metrics['rec'].keys():
                    metrics['rec'][epoch].append(history['recall'][epoch])
                else:
                    metrics['rec'][epoch] = [history['recall'][epoch]]

                if epoch in metrics['val_rec'].keys():
                    metrics['val_rec'][epoch].append(history['val_recall'][epoch])
                else:
                    metrics['val_rec'][epoch] = [history['val_recall'][epoch]]
            continue


            # loss values
            metrics['loss'].append(history['loss'])
            metrics['val_loss'].append(history['val_loss'])

            # auc values
            metrics['auc'].append(history['auc'])
            metrics['val_auc'].append(history['val_auc'])

            metrics['acc'].append(history['accuracy'])
            metrics['val_acc'].append(history['val_accuracy'])

            metrics['pre'].append(history['precision'])
            metrics['val_pre'].append(history['val_precision'])

            metrics['rec'].append(history['recall'])
            metrics['val_rec'].append(history['val_recall'])

    loss_tr_means, loss_val_means = analize_mean_std("loss", metrics, metrics)
    auc_tr_means, auc_val_means = analize_mean_std("auc", metrics, metrics)
    analize_mean_std("acc", metrics, metrics)
    analize_mean_std("pre", metrics, metrics)
    analize_mean_std("rec", metrics, metrics)


    loss_v_iteration_curve(500, loss_tr_means, loss_val_means, "Loss_vs_Iterations")

    auc_v_iterations_curve(500, auc_tr_means, auc_val_means, "AUCvsIterations")

    predictions_arr = []

    for file in os.listdir(predictions_dir):
        if os.path.isfile(os.path.join(predictions_dir, file)):
            file = open(os.path.join(predictions_dir, file), 'rb')
            predictions = pck.load(file)

            predictions_arr.append(predictions)

    plt.figure(2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PRE vs REC')
    plt.legend(loc='best')

    pre = []
    mean_rec = np.linspace(0, 1, 100)
    for idx, pred in enumerate(predictions_arr):
        pre.append(np.interp(mean_rec, pred['pre'], pred['rec']))

    mean_pre = np.mean(pre, axis=0)

    plt.plot(mean_pre, mean_rec)

    plt.savefig("PREvsREC_Project2CNN")
    plt.close()

    #ROCAUC
    plt.figure(3)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC AUC')


    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for idx, pred in enumerate(predictions_arr):

        plt.plot(pred['fpr'], pred['tpr'], 'r' ,alpha=0.15)

        interp_tpr = np.interp(mean_fpr, pred['fpr'], pred['tpr'])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(pred['auc'])

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc), lw=2, alpha=1)
    plt.legend(loc='best')
    plt.savefig("ROCAUC_Project2CNN.png")
    plt.close()

history_path = "./Histories/Histories_Model2"
predictions_path = "./Predictions/Predictions_Model2"
model_name = "Project2CNN"
epochs = 500

load_data_and_plot(history_path, predictions_path, epochs, model_name)









