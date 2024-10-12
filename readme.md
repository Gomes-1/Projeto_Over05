# Modelos de inteligência artificial para previsão de ocorrência de ao menos um gol no primeiro tempo de partidas do campeonato inglês de futebol (Premier League).


Este projeto utiliza uma rede neural para prever a ocorrência de ao menos um gol no primeiro tempo de partidas da Premier League. A rede é desenvolvida utilizando PyTorch e é treinada com dados estatísticos dos jogos.

## Descrição

O objetivo deste projeto é aplicar técnicas de deep learning para prever se haverá pelo menos um gol no primeiro tempo de uma partida de futebol da Premier League. Os dados utilizados incluem informações como finalizações, posse de bola e outras métricas importantes que influenciam o desempenho de cada equipe.

## Tecnologias Utilizadas

- **Linguagem de Programação:** Python
- **Bibliotecas:** 
  - `PyTorch`: Utilizado para construir e treinar a rede neural.
  - `pandas` e `numpy`: Para manipulação e processamento de dados.
  - `scikit-learn`: Para normalização dos dados.
  - `matplotlib`: Para visualização gráfica do desempenho do modelo.

## Estrutura do Projeto

- **`Net`**: Classe que define a arquitetura da rede neural, composta por camadas totalmente conectadas com funções de ativação ReLU e Sigmoid.
- **`redeNeural()`**: Função que treina a rede neural usando os dados fornecidos, calcula as perdas (erros) e avalia o desempenho.
- **`plotcharts()`**: Função para gerar gráficos de desempenho do modelo durante o treinamento, incluindo gráficos de erro e previsões.
- **`testar_modelo()`**: Função para testar o modelo já treinado em um novo conjunto de dados e calcular sua acurácia.

## Como Executar o Projeto

1. **Pré-requisitos:** Certifique-se de que você possui as seguintes bibliotecas instaladas:
   ```bash
   pip install pandas numpy torch matplotlib scikit-learn

### Treinamento da Rede Neural:

Para treinar a rede neural com um conjunto de dados específico, utilize a função `redeNeural()` passando os parâmetros necessários:

- hidden_size = 500
- redeNeural("Nome_do_Projeto", 0.8, 0.03, 20000, hidden_size, "premier-league-2023-2024-1")

### Estrutura dos Dados
- O modelo espera que os dados estejam em um arquivo CSV com as seguintes colunas principais:

- HomeTeam: Nome do time da casa.
- AwayTeam: Nome do time visitante.
- Over0.5: Indica se houve pelo menos um gol no primeiro tempo (1 para sim, 0 para não).
- Outras colunas incluem métricas como finalizações, posse de bola, chutes a gol, entre outras.
- Resultados Esperados
- O modelo será treinado para prever a variável Over0.5, que indica a ocorrência de pelo menos um gol no primeiro tempo. Acurácia de previsão esperada em torno de 65% (dependendo dos dados e do ajuste de hiperparâmetros).

### Visualizações
- Durante o treinamento, gráficos são gerados para:

- Erro de Treinamento: Mostra a evolução da perda ao longo das épocas.
- Previsões: Compara as previsões do modelo com os valores reais.
- Os gráficos são salvos como um arquivo chamado graficoTrain.png.

### Salvamento do Modelo
- O modelo treinado é salvo em um arquivo chamado modeloTreinado.pth, que pode ser carregado posteriormente para fazer novas previsões.

