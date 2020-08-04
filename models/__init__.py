import importlib
def find_model_from_string(model_name):
    modellib = importlib.import_module('models.%s' % model_name)
    class_name = model_name.replace('_', '')
    model = None
    for name, cls in modellib.__dict__.items():
        if name.lower() == class_name.lower():
            model = cls
            break
    if model is None:
        print('In %s.py, there should be a class named %s' % (model_name, class_name))
        exit(0)
    return model

def build_model(args, log):
    print('Creating Model %s' % (args.model))
    model_class = find_model_from_string(args.model)
    model = model_class(args, log)
    model.print_networks(log)
    return model

def get_option_setter(model_name):
    model_class = find_model_from_string(model_name)
    return model_class.modify_commandline_options
