from ..datasets.get_dataset import get_datasets
from ..recognition.get_model import get_model as get_rec_model
from ..models.get_model import get_model as get_gen_model


def get_model_and_data(parameters):
    datasets = get_datasets(parameters)

    print(datasets["train"].compositional_actions)
    parameters["compositional_actions"] = datasets["train"].compositional_actions
    print(parameters)
    print("uestc_actionmask", datasets["train"].human12_actionmask)
    parameters["human12_actionmask"] = datasets["train"].human12_actionmask

    if parameters["modelname"] == "recognition":
        model = get_rec_model(parameters)
    else:
        model = get_gen_model(parameters)
    return model, datasets
