python3.9 src/main.py method=T3_character_feedforward128_qEmb label_request=character model.num_classes=26 model.dim_feedforward=128
python3.9 src/main.py method=T3_style_feedforward128_qEmb label_request=style model.num_classes=4 model.dim_feedforward=128

# python3.9 src/main.py method=T3_character_feedforward256 label_request=character model.num_classes=26 model.dim_feedforward=256
# python3.9 src/main.py method=T3_style_feedforward256 label_request=style model.num_classes=4 model.dim_feedforward=256

# python3.9 src/main.py method=T3_character_feedforward512 label_request=character model.num_classes=26 model.dim_feedforward=512
# python3.9 src/main.py method=T3_style_feedforward512 label_request=style model.num_classes=4 model.dim_feedforward=512