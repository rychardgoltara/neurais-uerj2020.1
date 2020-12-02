# -*- coding: utf-8 -*-
"""
Created 
https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html#sphx-glr-download-auto-examples-neural-networks-plot-mlp-training-curves-py
"""

print(__doc__)

import numpy as np
import pandas as pd
from pandas import DataFrame

import warnings

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import validation_curve

import sklearn.metrics as metrics 

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

import warnings
warnings.filterwarnings("ignore")


# different learning rate schedules and momentum parameters
# o link abaixo apresenta a biblioteca para treinamento dos modelos.

#https://scikit-leaclass sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, ), 
#                                            activation='relu', *, solver='adam', 
#                                            alpha=0.0001, batch_size='auto', 
#                                            learning_rate='constant', 
# consulte os atributos para adequadamente ajustá-los.
#                                            learning_rate_init=0.001, power_t=0.5, 
#                                            max_iter=200, shuffle=True, 
#                                            random_state=None, tol=0.0001, 
#                                            verbose=False, warm_start=False, 
#                                            momentum=0.9, nesterovs_momentum=True, 
#                                            early_stopping=False, 
#                                            validation_fraction=0.1, beta_1=0.9, 
#                                            beta_2=0.999, epsilon=1e-08, 
#                                            n_iter_no_change=10, max_fun=15000)rn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html



# Exemplo de parametrizaçao
# parametros a ajustar: solver (sgd, adam), learning_rate, momentum
 
params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0.8,
           'learning_rate_init': 0.4},
              {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0.9,
                'learning_rate_init': 0.5},
          {'solver': 'sgd', 'learning_rate': 'adaptive', 'momentum': 0.9,
            'nesterovs_momentum': True, 'learning_rate_init': 0.4},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
            'learning_rate_init': 0.4},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0.9,
             'nesterovs_momentum': True, 'learning_rate_init': 0.4},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0.9,
             'nesterovs_momentum': True, 'learning_rate_init': 0.5},
          {'solver': 'adam', 'learning_rate_init': 0.01,'beta_1':0.85, 'beta_2': 0.9},
          {'solver': 'adam', 'learning_rate_init': 0.02,'beta_1':0.85, 'beta_2': 0.9},
          {'solver': 'adam', 'learning_rate_init': 0.05,'epsilon': 0.00001, 'beta_1':0.9, 'beta_2': 0.99},
          {'solver': 'adam', 'learning_rate_init': 0.2,'epsilon': 0.01, 'beta_1':0.7, 'beta_2': 0.9},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0.7,
           'learning_rate_init': 0.2}, 
          {'solver': 'sgd', 'learning_rate': 'adaptive', 'momentum': 0.5,
            'nesterovs_momentum': True, 'learning_rate_init': 0.1},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0.7,
             'nesterovs_momentum': True, 'learning_rate_init': 0.25},
          {'solver': 'lbfgs', 'max_fun': 15000},
           {'solver': 'lbfgs', 'max_fun': 20000},
           {'solver': 'lbfgs', 'max_fun': 10000},
           {'solver': 'lbfgs', 'max_fun': 5000}]


#indique abaixo a parametrizacao apenas para registro dos resultados

labels = [
           "1 sgd constant learning-rate with momentum=0.8 lear_init =0.4",
           "2 sgd constant learning-rate with momentum=0.9 lear_init =0.5",
           "3 sgd adaptative learning-rate with momentum=0.9 lear_init =0.4 nesterovs = True",
           "4 sgd invscaling learning-rate with momentum=0 lear_init =0.4 ",
           "5 sgd invscaling learning-rate with momentum=0.9 lear_init =0.4 nesterovs = True",
           "6 sgd invscaling learning-rate with momentum=0.9 lear_init =0.5 nesterovs = True",
           "7 adam rate_init=0.01 beta_1=0.85  beta_2=0.9",
           "8 adam rate_init=0.02  beta_1=0.85  beta_2=0.9",
           "9 adam rate_init=0.05 epsilon=0.00001 beta_1=0.9  beta_2=0.99",
           "10 adam rate_init=0.2 epsilon=0.01 beta_1=0.7  beta_2=0.9",
           "11 sgd constant learning-rate with momentum=0.7 lear_init =0.2",
           "12 sgd adaptative learning-rate with momentum=0.5 lear_init =0.1 nesterovs = True",
           "13 sgd invscaling learning-rate with momentum=0.7 lear_init =0.25 nesterovs = True",
           "14 lbfgs max_fun = 15000",
           "15 lbfgs max_fun = 20000",
           "16, lbfgs max_fun = 10000",
           "17, lbfgs max_fun = 5000"
            ]



def busca_ajuste_modelos(ds, name):
    # for each dataset, plot learning for each learning strategy
    file1 = open("resultados.txt","a")
    mlps = []
    mlps_best=[]
    mlps_scores=[]
    mlps_scores_best=[]
    file1.write ("\nLearning on dataset %s" % name)
    print("\nLearning on dataset %s" % name)
    file1.write("\n Epochs = 200, repeats = 3, Neur = 10, Batch = 200  activation =identity \n")
    
    X,y=ds[0]
    
    # defina o número de epochs
    epochs = 200
    max_iter = epochs   
    
    arch_mean_acc=[]
    arch_mean_prec=[]
    arch_mean_rec=[]
    arch_mean_f1=[]               
                    
    
    for label, param in zip(labels, params):
        print('----------------------------------------')
        file1.write ("---------------------------------------")
  
        file1.write ("\n Training a arquitetura: %s \n" % label)
        print("Training a arquitetura: %s" % label)
        print()
        file1.write("\n")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                    module="sklearn")
            repeats = 3
            
            N=11 # Número máximo de neuronios a serem avaliados na camada escondida
            for nh in range(1,N):
                hidden_sizes=(nh, )
                
                acc_array=[]
                rec_array=[]
                prec_array=[]
                f1_array=[]
                
                # defina o número de instancias a serem ava
                for instancias in range(0,repeats):
     
                # para executar depois em função do número de neurônios
                # da camada escondida a melhor arquitetura considerando a
                # score medio das n instancias criadas
                
                # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
                    #scoring= ['precision', 'recall', 'accuracy', 'f1']
                    # para executar depois de avaliar os modelos pelo score medio
                    scoring= ['precision', 'recall', 'accuracy', 'f1']

                    mlp = MLPClassifier(random_state=0, max_iter=max_iter, **param,
                                        hidden_layer_sizes= hidden_sizes, activation='identity', batch_size = 200, early_stopping=True, n_iter_no_change=10)
                                        
                    scores = cross_validate(mlp, X, y, cv=8, scoring=scoring, return_estimator=True, return_train_score=True)
                   
                    # cria o vetor com as medias dos scores para esse NH para 
                    # cada uma das instäncias
                    
                    scor_array=scores['test_recall']
                    rec_array.append(scor_array)
                                
                    scor_array=scores['test_accuracy']
                    acc_array.append(scor_array)
            
                    scor_array=scores['test_precision']
                    prec_array.append(scor_array)
    
                    scor_array=scores['test_f1']
                    f1_array.append(scor_array)
                

                    
                
                # valor medio das metricas para o número de instâncias
                mean_model_acc=np.mean(acc_array)
                mean_model_prec=np.mean(prec_array)
                mean_model_rec=np.mean(rec_array)
                mean_model_f1=np.mean(f1_array)
                print('Erro médio de validação cruzada com %d HN' % nh)
                file1.write('Erro médio de validação cruzada com %d HN \n' % nh)
                print("accuracy score: %2.4f" % mean_model_acc)
                file1.write("accuracy score: %2.4f \n" % mean_model_acc)
                print("recall score: %2.4f" % mean_model_rec)
                file1.write("recall score: %2.4f \n" % mean_model_rec)
                print("precision score: %2.4f" % mean_model_prec)
                file1.write("precision score: %2.4f \n" % mean_model_prec)
                print("f1 score: %2.4f" % mean_model_f1)
                file1.write("f1 score: %2.4f \n" % mean_model_f1)
                print()
                file1.write("\n")

                # arch_mean_acc.append(mean_model_acc)
                # arch_mean_prec.append(mean_model_prec)
                # arch_mean_rec.append(mean_model_rec)
                # arch_mean_f1.append(mean_model_f1)
                    
            
                
            # for i in range(0,len(arch_mean_acc)):   
                
            #     # indice para a arquitetura da melhor acuracia para essa parametrizacao            
            #     ind = np.argmax(arch_mean_rec)
            #     print('melhor resultado para esta arquitetura')
            #     print('Número de HN = %d' % ind)
            #     print('accuracy  = %2.4f' % arch_mean_acc[ind])
            #     print('recall  = %2.4f' % arch_mean_rec[ind])
            #     print('precision = %2.4f' % arch_mean_prec[ind])
            #     print('F1 = %2.4f.' % arch_mean_f1[ind])
            #     print('----------------------------------------')
            #     print()
            #     print()
    
    file1.close()
    
    
def busca_ajuste_melhor_resultado(ds, name):
    
    # for each dataset, plot learning for each learning strategy
    
    print("\nlearning on dataset %s" % name)
    
    X,y=ds[0]
    #X = MinMaxScaler().fit_transform(X)
    max_iter=1
        
    acc_array=[]
    rec_array=[]
    prec_array=[]
    f1_array=[]
    
    mlps=[]     
    #           
    #indicar os parâmetros do modelo escolhido como o melhor
    #
    params_best = [
          {'solver': 'adam', 'learning_rate_init': 0.2,'epsilon': 0.01, 'beta_1':0.7, 'beta_2': 0.9}] 

    label_best = [
            "10 adam rate_init=0.2 epsilon=0.01 beta_1=0.7  beta_2=0.9"]       
                
    
    # Definir número de NH
    best_NH=6
    #
    hidden_sizes = (best_NH, )  
  
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                module="sklearn")
        for label, params in zip(label_best, params_best):
            
            
            # defina o número de epochs
            epochs = 200
            max_iterint = epochs

            # defina o número de instancias a serem avaliadas
            repeats = 3

            for i in range(0,repeats):
                print("training: %s" % label)
    
                mlp = MLPClassifier(random_state=0, max_iter=max_iterint, **params, 
                                    hidden_layer_sizes= hidden_sizes, activation='relu', batch_size= 200, early_stopping=True, 
                                    validation_fraction=0.2, n_iter_no_change=5)
    
               # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
                scoring= ['precision', 'recall', 'accuracy', 'f1']
                # para executar depois de avaliar os modelos pelo score medio
                #scores = mlp.fit(X, y) 
                scores = cross_validate(mlp, X, y, cv=5, scoring=scoring, return_estimator=True, return_train_score=True)

                
                # param_range=np.arange(1,5)
                # train_scores, test_scores = validation_curve(mlp,X, y, cv=5, param_name="hidden_layer_sizes , param_range=param_range, scoring="accuracy",)
                               
                # train_scores_mean = np.mean(train_scores, axis=1)
                # train_scores_std = np.std(train_scores, axis=1)
                # test_scores_mean = np.mean(test_scores, axis=1)
                # test_scores_std = np.std(test_scores, axis=1)                      
                
                # plt.title("Validation Curve with SVM")
                # plt.xlabel(r"$\gamma$")
                # plt.ylabel("Score")
                # plt.ylim(0.0, 1.1)
                # lw = 2
                # plt.semilogx(param_range, train_scores_mean, label="Training score",
                #              color="darkorange", lw=lw)
                # plt.fill_between(param_range, train_scores_mean - train_scores_std,
                #                  train_scores_mean + train_scores_std, alpha=0.2,
                #                  color="darkorange", lw=lw)
                # plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                #              color="navy", lw=lw)
                # plt.fill_between(param_range, test_scores_mean - test_scores_std,
                #                  test_scores_mean + test_scores_std, alpha=0.2,
                #                  color="navy", lw=lw)
                # plt.legend(loc="best")
                # plt.show()

                scor_array=scores['test_recall']
                rec_array.append(np.mean(scor_array))
                          
                scor_array=scores['test_accuracy']
                acc_array.append(np.mean(scor_array))
                
                #armazena o melhor modelo (para acuracia) entre os n do cross validation 
                ind_best_repeat = np.argmax(acc_array)
                
        
                scor_array=scores['test_precision']
                prec_array.append(np.mean(scor_array))

                scor_array=scores['test_f1']
                f1_array.append(np.mean(scor_array))                
                
                
                scor_array=scores['estimator']         
                
                mlps.append(scor_array[ind_best_repeat])               


                
                # y_pred=mlp.predict(X)
                
                # scor_array=recall_score(y,y_pred)
                # rec_array.append(np.mean(scor_array))
                                    
                # scor_array=accuracy_score(y,y_pred)
                # acc_array.append(np.mean(scor_array))
                
                # scor_array=precision_score(y,y_pred)
                # prec_array.append(np.mean(scor_array))
    
                # scor_array=f1_score(y,y_pred)
                # f1_array.append(np.mean(scor_array))
                
               
                
        # #retorna indice relativo ao modelo de melhor acurácia 
        # dentre os n avaliados pelo cross valitation com a melhor acurácia média
        ind_best = np.argmax(acc_array)
   
    
        print('Erro validação')
        print("accuracy score: %2.4f" % acc_array[ind_best])
        print("recall score: %2.4f" % rec_array[ind_best])
        print("precision score: %2.4f" % prec_array[ind_best])
        print("f1 score: %2.4f" % f1_array[ind_best])
                
        # forecast dataset
        
        X,y=ds[1]

        y_test_pred=(mlps[ind_best].predict(X))
        
        # invert data transforms on forecast
        acc=accuracy_score(y, y_test_pred)
        precision=precision_score(y, y_test_pred)
        f1=f1_score(y, y_test_pred)    	
        recall=recall_score(y, y_test_pred)
        cm=confusion_matrix(y, y_test_pred)
        
        print("RESULTADOS TESTE")
        print("accuracy", acc)
        print("precision", precision)
        print("recall", recall)    
        print("f1", f1)
        print("cm", cm)
        
        np.set_printoptions(precision=2)
        class_names=[]
        class_names.append("CONVID-19")
        class_names.append("Não CONVID-19")
    
        # Plot non-normalized confusion matrix
        titles_options = [("Confusion matrix, without normalization", None),
                          ("Normalized confusion matrix", 'true')]
        for title, normalize in titles_options:
            disp = plot_confusion_matrix( mlps[ind_best], X, y,
                                          display_labels=class_names,
                                          cmap=plt.cm.Blues,
                                          normalize=normalize)
            disp.ax_.set_title(title)
        
            print(title)
            print(disp.confusion_matrix)
        
        plt.show() 
             
    
#########################################################
# leitura de dados UX
#planilha_1=pd.read_excel(r"C:\Users\Karla\Dropbox\UERJ\IME\Projetos\2020\Experiencia do Usuário\Base 2020-17-Abril\Dados UX-SEPARADOS.xlsx",sheet_name=0)
planilha_1=pd.read_excel(r"C:\Users\rycba\OneDrive\Documentos\Classificacao Covid\base_pacientes_sintomas.xlsx",sheet_name="treino")


X = DataFrame(planilha_1) 
X=np.matrix(X)  
    
# fit scaler
# X = StandardScaler().fit_transform(X)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(X)
    
# transform train
X = X.reshape(X.shape[0], X.shape[1])
#X_scaled = scaler.transform(X)
    
# transform 
#X=X_scaled

X_train = X[:,0:11]
y_train = X[:,11]

#planilha_1=pd.read_excel(r"C:\Users\Karla\Dropbox\UERJ\IME\Projetos\2020\Experiencia do Usuário\Base 2020-17-Abril\Dados UX-SEPARADOS.xlsx",sheet_name="teste")
planilha_1=pd.read_excel(r"C:\Users\rycba\OneDrive\Documentos\Classificacao Covid\base_pacientes_sintomas.xlsx",sheet_name="teste")
X_teste = DataFrame(planilha_1) 
X_teste=np.matrix(X_teste)  
X_teste = X_teste.reshape(X_teste.shape[0], X_teste.shape[1])

#X_teste_scaled = scaler.transform(X_teste)
# transform 
#X_teste=X_teste_scaled

X_test = X_teste[:,0:11]
y_test = X_teste[:,11]


#################################
data_sets = [(X_train, y_train),(X_test, y_test)]
name=['COVID-19']

# função para avaliar a melhor arquitetura e parâmetros para escolha do melhor modelo
#busca_ajuste_modelos(data_sets, name=name)


# função para avaliar os resultados de teste após a escolha do melhor modelo
busca_ajuste_melhor_resultado(data_sets, name=name)



