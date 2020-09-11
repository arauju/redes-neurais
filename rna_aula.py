import matplotlib.pyplot as plt
import numpy as np
import math

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.structure import SigmoidLayer
from pybrain.structure import SoftmaxLayer
from pybrain.structure import LinearLayer
from pybrain.structure import MDLSTMLayer
from pybrain.structure import LSTMLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.validation import ModuleValidator

if __name__ == '__main__':

    # Definicao da classe de rede neural
    dimensaoDaEntrada=1
    dimensaoDaCamadaEscondida=22
    dimensaoDaSaida=1

    rn=buildNetwork(dimensaoDaEntrada,dimensaoDaCamadaEscondida,dimensaoDaSaida,bias=True,hiddenclass=TanhLayer)

    #Criacao dos dados
    tamanhoDaAmostra=400
    dados = SupervisedDataSet(dimensaoDaEntrada,dimensaoDaSaida)

    comRuido=False

    #Gera uma amostra da funcao f(x) = sen(x)
    for i in range(tamanhoDaAmostra):
        if(comRuido):
            x=np.random.uniform(0,2*math.pi,1)
            dados.addSample((x), (math.sin(x)+ np.random.normal(0, 0.1,1),))
        else:
            x=np.random.uniform(0,2*math.pi,1)
            dados.addSample((x), (math.sin(x),))

    treinoSupervisionado = BackpropTrainer(rn, dados)

    # quantidade de iteracoes. Saidas canalizadas nas entradas da RNA
    numeroDeIteracoes=120 
    # a apresentacao de todos os padroes de treinamento disponiveis corresponde a uma epoca
    numeroDeEpocasPorIteracao=180 


    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.axis([0, 2*math.pi, -1.5, 1.5])


    fig2=plt.figure()
    ax2 = fig2.add_subplot(111)
    #desenha as coordenadas em Y com escala logaritima
    ax2.axis([-50, numeroDeIteracoes*numeroDeEpocasPorIteracao+50, 0.0000001, 4])
    ax2.set_yscale('log')

    meansq = ModuleValidator()
    erro2=meansq.MSE(treinoSupervisionado.module,dados)
    print erro2
    ax2.plot([0],[erro2],'bo')

    tempoPausa=1
    for i in range(numeroDeIteracoes):
        treinoSupervisionado.trainEpochs(numeroDeEpocasPorIteracao)
        meansq = ModuleValidator()
        erro2=meansq.MSE(treinoSupervisionado.module,dados)
        print erro2
	#Desenha a linha azul com o valor padrao
        ax1.plot(dados['input'],dados['target'],'bo',markersize=7, markeredgewidth=0)
        ax1.plot(dados['input'],np.array([rn.activate(datax) for datax, _ in dados]),'ro',markersize=7, markeredgewidth=0)
        ax2.plot([numeroDeEpocasPorIteracao*(i+1)],[erro2],'bo')
        plt.pause(tempoPausa)
	#Desenha com pontos brancos maiores sobre os vermelhos.
        ax1.plot(dados['input'],np.array([rn.activate(datax) for datax, _ in dados]),'wo',markersize=9, markeredgewidth=0)
