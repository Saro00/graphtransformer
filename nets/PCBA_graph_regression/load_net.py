"""
    Utility file to select GraphNN model as
    selected by the user
"""
from nets.PCBA_graph_regression.graph_transformer_net import GraphTransformerNet


def GraphTransformer(net_params):
    return GraphTransformerNet(net_params)


def gnn_model(MODEL_NAME, net_params):
    models = {
        'GraphTransformer': GraphTransformer
    }

    return models[MODEL_NAME](net_params)