########################### ws用 ###########################
# python src/main.py method=ViT_googlefonts_character batch_size=16 model=ViT label_request=character lim=2838 model.num_classes=26 _fold=0
# python src/main.py method=ViT_googlefonts_character batch_size=16 model=ViT label_request=character lim=2838 model.num_classes=26 _fold=1
# python src/main.py method=ViT_googlefonts_character batch_size=16 model=ViT label_request=character lim=2838 model.num_classes=26 _fold=2
# python src/main.py method=ViT_googlefonts_character batch_size=16 model=ViT label_request=character lim=2838 model.num_classes=26 _fold=3
# python src/main.py method=ViT_googlefonts_character batch_size=16 model=ViT label_request=character lim=2838 model.num_classes=26 _fold=4
# python src/main.py method=ViT_googlefonts_style batch_size=16 model=ViT label_request=style lim=2838 model.num_classes=4 _fold=0
# python src/main.py method=ViT_googlefonts_style batch_size=16 model=ViT label_request=style lim=2838 model.num_classes=4 _fold=1
# python src/main.py method=ViT_googlefonts_style batch_size=16 model=ViT label_request=style lim=2838 model.num_classes=4 _fold=2
# python src/main.py method=ViT_googlefonts_style batch_size=16 model=ViT label_request=style lim=2838 model.num_classes=4 _fold=3
# python src/main.py method=ViT_googlefonts_style batch_size=16 model=ViT label_request=style lim=2838 model.num_classes=4 _fold=4
# python src/main.py method=ViT_googlefonts_character batch_size=16 model=ViT label_request=character lim=100 model.num_classes=26 _fold=0
# python src/main.py method=ViT_googlefonts_character batch_size=16 model=ViT label_request=character lim=100 model.num_classes=26 _fold=1
# python src/main.py method=ViT_googlefonts_character batch_size=16 model=ViT label_request=character lim=100 model.num_classes=26 _fold=2
# python src/main.py method=ViT_googlefonts_character batch_size=16 model=ViT label_request=character lim=100 model.num_classes=26 _fold=3
# python src/main.py method=ViT_googlefonts_character batch_size=16 model=ViT label_request=character lim=100 model.num_classes=26 _fold=4
# python src/main.py method=ViT_googlefonts_style batch_size=16 model=ViT label_request=style lim=100 model.num_classes=4 _fold=0
# python src/main.py method=ViT_googlefonts_style batch_size=16 model=ViT label_request=style lim=100 model.num_classes=4 _fold=1
# python src/main.py method=ViT_googlefonts_style batch_size=16 model=ViT label_request=style lim=100 model.num_classes=4 _fold=2
# python src/main.py method=ViT_googlefonts_style batch_size=16 model=ViT label_request=style lim=100 model.num_classes=4 _fold=3
# python src/main.py method=ViT_googlefonts_style batch_size=16 model=ViT label_request=style lim=100 model.num_classes=4 _fold=4

# python src/main_CNN2D.py method=Res_googlefonts_character_16 batch_size=16 model=CNN2D model.img_size=16 model.num_classes=26 label_request=character
# python src/main_CNN2D.py method=Res_googlefonts_character_32 batch_size=16 model=CNN2D model.img_size=32 model.num_classes=26 label_request=character
# python src/main_CNN2D.py method=Res_googlefonts_character_64 batch_size=16 model=CNN2D model.img_size=64 model.num_classes=26 label_request=character
# python src/main_CNN2D.py method=Res_googlefonts_character_128 batch_size=16 model=CNN2D model.img_size=128 model.num_classes=26 label_request=character
# python src/main_CNN2D.py method=Res_googlefonts_character_256 batch_size=16 model=CNN2D model.img_size=256 model.num_classes=26 label_request=character

# python src/main_CNN2D.py method=Res_googlefonts_style_16 batch_size=16 model=CNN2D model.img_size=16 model.num_classes=4 label_request=style
# python src/main_CNN2D.py method=Res_googlefonts_style_32 batch_size=16 model=CNN2D model.img_size=32 model.num_classes=4 label_request=style
# python src/main_CNN2D.py method=Res_googlefonts_style_64 batch_size=16 model=CNN2D model.img_size=64 model.num_classes=4 label_request=style
# python src/main_CNN2D.py method=Res_googlefonts_style_128 batch_size=16 model=CNN2D model.img_size=128 model.num_classes=4 label_request=style
# python src/main_CNN2D.py method=Res_googlefonts_style_256 batch_size=16 model=CNN2D model.img_size=256 model.num_classes=4 label_request=style

########################### local用 ###########################

/home/yusuke/anaconda3/envs/tf/bin/python /home/yusuke/Dev/Outline/src/main.py method=ViT_googlefonts_character model=ViT label_request=character lim=2838 model.num_classes=26
# /home/yusuke/anaconda3/envs/tf/bin/python /home/yusuke/Dev/Outline/src/main.py method=ViT_googlefonts_style model=ViT label_request=style lim=2838 model.num_classes=4
# /home/yusuke/anaconda3/envs/tf/bin/python /home/yusuke/Dev/Outline/src/main.py method=ViT_googlefonts_character_lim100 model=ViT label_request=character lim=100 model.num_classes=26
# /home/yusuke/anaconda3/envs/tf/bin/python /home/yusuke/Dev/Outline/src/main.py method=ViT_googlefonts_style_lim100 model=ViT label_request=style lim=100 model.num_classes=4

# /home/yusuke/anaconda3/envs/tf/bin/python /home/yusuke/Dev/Outline/src/main_CNN2D.py method=Res_googlefonts_character_16 batch_size=512 model=CNN2D model.img_size=16 model.num_classes=26 label_request=character
# /home/yusuke/anaconda3/envs/tf/bin/python /home/yusuke/Dev/Outline/src/main_CNN2D.py method=Res_googlefonts_character_32 batch_size=256 model=CNN2D model.img_size=32 model.num_classes=26 label_request=character
# /home/yusuke/anaconda3/envs/tf/bin/python /home/yusuke/Dev/Outline/src/main_CNN2D.py method=Res_googlefonts_character_64 batch_size=256 model=CNN2D model.img_size=64 model.num_classes=26 label_request=character
# /home/yusuke/anaconda3/envs/tf/bin/python /home/yusuke/Dev/Outline/src/main_CNN2D.py method=Res_googlefonts_character_128 batch_size=128 model=CNN2D model.img_size=128 model.num_classes=26 label_request=character
# /home/yusuke/anaconda3/envs/tf/bin/python /home/yusuke/Dev/Outline/src/main_CNN2D.py method=Res_googlefonts_character_256 batch_size=128 model=CNN2D model.img_size=256 model.num_classes=26 label_request=character

# /home/yusuke/anaconda3/envs/tf/bin/python /home/yusuke/Dev/Outline/src/main_CNN2D.py method=Res_googlefonts_style_16 batch_size=512 model=CNN2D model.img_size=16 model.num_classes=4 label_request=style
# /home/yusuke/anaconda3/envs/tf/bin/python /home/yusuke/Dev/Outline/src/main_CNN2D.py method=Res_googlefonts_style_32 batch_size=256 model=CNN2D model.img_size=32 model.num_classes=4 label_request=style
# /home/yusuke/anaconda3/envs/tf/bin/python /home/yusuke/Dev/Outline/src/main_CNN2D.py method=Res_googlefonts_style_64 batch_size=256 model=CNN2D model.img_size=64 model.num_classes=4 label_request=style
# /home/yusuke/anaconda3/envs/tf/bin/python /home/yusuke/Dev/Outline/src/main_CNN2D.py method=Res_googlefonts_style_128 batch_size=128 model=CNN2D model.img_size=128 model.num_classes=4 label_request=style
# /home/yusuke/anaconda3/envs/tf/bin/python /home/yusuke/Dev/Outline/src/main_CNN2D.py method=Res_googlefonts_style_256 batch_size=128 model=CNN2D model.img_size=256 model.num_classes=4 label_request=style


##########otao##############################
# python src/main_CNN2D.py method=otao_googlefonts_character_16 batch_size=512 model=vitotao model.img_size=16 model.num_classes=26 label_request=character
# python src/main_CNN2D.py method=otao_googlefonts_character_32 batch_size=256 model=vitotao model.img_size=32 model.num_classes=26 label_request=character
# python src/main_CNN2D.py method=otao_googlefonts_character_64 batch_size=256 model=vitotao model.img_size=64 model.num_classes=26 label_request=character
# python src/main_CNN2D.py method=otao_googlefonts_character_128 batch_size=128 model=vitotao model.img_size=128 model.num_classes=26 label_request=character
# python src/main_CNN2D.py method=otao_googlefonts_character_256 batch_size=128 model=vitotao model.img_size=256 model.num_classes=26 label_request=character

# python src/main_CNN2D.py method=otao_googlefonts_style_16 batch_size=512 model=vitotao model.img_size=16 model.num_classes=4 label_request=style
# python src/main_CNN2D.py method=otao_googlefonts_style_32 batch_size=256 model=vitotao model.img_size=32 model.num_classes=4 label_request=style
# python src/main_CNN2D.py method=otao_googlefonts_style_64 batch_size=256 model=vitotao model.img_size=64 model.num_classes=4 label_request=style
# python src/main_CNN2D.py method=otao_googlefonts_style_128 batch_size=128 model=vitotao model.img_size=128 model.num_classes=4 label_request=style
# python src/main_CNN2D.py method=otao_googlefonts_style_256 batch_size=128 model=vitotao model.img_size=256 model.num_classes=4 label_request=style