from utils import split_prep, convert_to_tf

if __name__ == "__main__":

    final_architecture = {
        "normal_n2_p0": "maxpool", 
        "normal_n2_p1": "sepconv3x3", 
        "normal_n3_p0": "maxpool", 
        "normal_n3_p1": "dilconv3x3", 
        "normal_n3_p2": "skipconnect",
        "normal_n4_p0": "dilconv3x3",
        "normal_n4_p1": "dilconv3x3", 
        "normal_n4_p2": "skipconnect", 
        "normal_n4_p3": "skipconnect", 
        "normal_n5_p0": "dilconv5x5", 
        "normal_n5_p1": "dilconv5x5", 
        "normal_n5_p2": "skipconnect", 
        "normal_n5_p3": "skipconnect", 
        "normal_n5_p4": "skipconnect", 
        "reduce_n2_p0": "maxpool", 
        "reduce_n2_p1": "maxpool", 
        "reduce_n3_p0": "maxpool", 
        "reduce_n3_p1": "maxpool", 
        "reduce_n3_p2": "dilconv5x5", 
        "reduce_n4_p0": "maxpool", 
        "reduce_n4_p1": "maxpool", 
        "reduce_n4_p2": "skipconnect", 
        "reduce_n4_p3": "skipconnect", 
        "reduce_n5_p0": "sepconv5x5", 
        "reduce_n5_p1": "maxpool", 
        "reduce_n5_p2": "skipconnect", 
        "reduce_n5_p3": "skipconnect", 
        "reduce_n5_p4": "skipconnect", 
        "normal_n2_switch": [0, 1],
        "normal_n3_switch": [0, 1], 
        "normal_n4_switch": [0, 1], 
        "normal_n5_switch": [0, 2], 
        "reduce_n2_switch": [1, 0], 
        "reduce_n3_switch": [0, 1], 
        "reduce_n4_switch": [1, 0], 
        "reduce_n5_switch": [0, 1]
        }

    dict_normal, dict_reduce = split_prep(final_architecture)

    convert_to_tf(
        dict_normal=dict_normal, 
        dict_reduce=dict_reduce, 
        layers=8, 
        channels=16, 
        filename="architectures/layer8_epochs15_batch32_channels16"
        )
