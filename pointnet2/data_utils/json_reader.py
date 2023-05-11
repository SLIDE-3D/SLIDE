import json
import argparse
import os
import copy
import pdb

def replace_list_with_string_in_a_dict(dictionary):
    # dict_keys = []
    for key in dictionary.keys():
        if isinstance(dictionary[key], list):
            dictionary[key] = str(dictionary[key])
        if isinstance(dictionary[key], dict):
            dictionary[key] = replace_list_with_string_in_a_dict(dictionary[key])
    return dictionary

def restore_string_to_list_in_a_dict(dictionary):
    for key in dictionary.keys():
        try:
            evaluated = eval(dictionary[key])
            if isinstance(evaluated, list):
                dictionary[key] = evaluated
        except:
            pass
        if isinstance(dictionary[key], dict):
            dictionary[key] = restore_string_to_list_in_a_dict(dictionary[key])
    return dictionary

def read_json_file(config_file):
    with open(config_file) as f:
        data = f.read()
    config = json.loads(data)
    config = restore_string_to_list_in_a_dict(config)
    return config

def autoencoder_read_config(config_dir, config):
    encoder_config_file = os.path.join(config_dir, config['pointnet_config']['encoder_config_file'])
    decoder_config_file = [os.path.join(config_dir, config_i) for config_i in config['pointnet_config']['decoder_config_file']]

    encoder_config = read_json_file(encoder_config_file)['pointnet_config']
    decoder_config_list = []
    for decoder_config in decoder_config_file:
        decoder_config_list.append(read_json_file(decoder_config)['pointnet_config'])
    return encoder_config, decoder_config_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json', 
                        help='JSON file for configuration')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    # pdb.set_trace()
    config_string = replace_list_with_string_in_a_dict(copy.deepcopy(config))
    print('The configuration is:')
    print(json.dumps(config_string, indent=4))

    config_restore = restore_string_to_list_in_a_dict(config_string)
    print(config_restore == config)
    pdb.set_trace()
