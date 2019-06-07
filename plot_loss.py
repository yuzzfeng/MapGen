import numpy as np
import matplotlib.pyplot as plt

def plot_compare(paths, path_names):
    new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
    
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    
    count = 0
    for path_acc, path_name in zip(paths, path_names):
        fn_train_acc = "History1_acc.txt"
        fn_valid_acc = "History1_val_acc.txt"
        
        acc_train_values = np.loadtxt(path_acc + fn_train_acc)
        acc_valid_values = np.loadtxt(path_acc + fn_valid_acc)
        
        acc_train_values = (np.ones(len(acc_train_values)) - acc_train_values) * 100
        acc_valid_values = (np.ones(len(acc_valid_values)) - acc_valid_values) * 100
        
        color = new_colors[count]
    
        #l1, = ax.plot(acc_train_values, c = color, dashes=[6, 2])
        l2, = ax.plot(acc_valid_values, c = color, label=path_name)
    
        count += 1
    
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error (%)')
    plt.show()
    
###################### Select loss Adam(0.0001, 0.5) ######################

## 1:25k
#paths = ["2018-07-20 07-53-50_25/", 
#         #"2019-02-20 12-55-43_25/", 
#         #"2019-02-21 09-43-28_25/", 
#         "2019-02-20 15-20-05_25/", 
#         "2019-02-21 12-14-20_25/",
#         "2019-02-20 19-05-25_25/", 
#         "2019-02-25 08-45-18_25/",
#         #"U128_2019-03-11 21-37-36_25/"
#         ]
#
#path_names = ['U-net', 
#              #'Residual U-net with L2 Norm old', 
#              #"Residual U-net L2 RMSprop", 
#              'Residual U-net with L1', 
#              'Residual U-net with L2', 
#              'Residual U-net with Cross-entropy', 
#              'Residual U-net with DICE',
#              #'Residual U-net with Focal', 
#              ]
#plot_compare(paths, path_names)
#
## 1:15k
#paths = ["2018-07-19 13-13-22_15/", 
#         #"2019-02-21 14-40-38_15/", 
#         "2019-02-22 16-12-59_15/",
#         "2019-02-25 12-49-50_15/",
#         "2019-02-22 14-01-55_15/", 
#         "2019-02-24 17-27-11_15/", 
#         #"U128_2019-03-11 19-12-58_15/"
#         ]
#path_names = ['U-net', 
#              #"Residual U-net with L2 Norm old", 
#              'Residual U-net with L1', 
#              "Residual U-net with L2",
#              'Residual U-net with Cross-entropy',
#              'Residual U-net with DICE', 
#              #'Residual U-net with Focal', 
#              ]
#plot_compare(paths, path_names)
#
## 1:10k
#paths = ["2018-07-19 15-14-38_10/", 
#         "2019-02-22 09-46-24_10/", 
#         "2019-02-21 16-24-04_10/", 
#         "2019-02-22 12-41-40_10/",
#         "2019-02-25 10-28-15_10/"]
#path_names = ['U-net', 
#              "Residual U-net with L1",
#              "Residual U-net with L2", 
#              'Residual U-net with Cross-entropy', 
#              'Residual U-net with DICE']
#plot_compare(paths, path_names)

###################### Select optimizer L2 ######################

### 1:25k for optimizer
#paths = ["2018-07-20 07-53-50_25/", 
#         "2019-03-01 16-17-09_25/",
#         "2019-02-20 12-55-43_25/",  
#         "2019-02-28 11-05-31_25/", 
#         "2019-02-28 13-20-22_25/",
#         "2019-02-21 09-43-28_25/",
#         #"2019-02-21 12-14-20_25/", 
#         #"2019-03-03 14-09-56_25/",
#         ]
#
#path_names = ['U-net', 
#              "Residual U-net with Adadelta", # lr=1.0
#              'Residual U-net with Adam lr=0.0001', #L2 Norm old Adam(0.0001, 0.5) 
#              "Residual U-net with Adam lr=0.0004", # Current Best!!!
#              "Residual U-net with Adam lr=0.001",
#              "Residual U-net with RMSprop lr=0.001",
#              #'Residual U-net with L2 Norm Adam(0.0001, 0.5)', 
#              #"Adadelta(0.0001)", # Worst
#              ]
#plot_compare(paths, path_names)
#
## 1:15k for optimizer
#paths = ["2018-07-19 13-13-22_15/", 
#         "U128_2019-03-11 16-08-17_15/",
#         "2019-02-26 15-35-51_15/", 
#         "2019-02-26 12-55-00_15/",
#         "2019-02-25 15-55-01_15/", 
#         "2019-02-26 08-44-29_15/", 
#         #"2019-02-25 12-49-50_15/",
#         #"2019-02-27 09-29-08_15/",
#         #"2019-02-27 13-13-37_15/", 
#         #"2019-02-27 15-26-11_15/",
#         #"2019-02-26 10-30-03_15/",
#         #"2019-03-07 13-14-06_15/",
#         #"2019-03-07 15-06-54_15/"
#         ]
#
#path_names = ['U-net', 
#              "Residual U-net with Adadelta",
#              "Residual U-net with Adam lr=0.0001", 
#              "Residual U-net with Adam lr=0.0004", # Current best
#              "Residual U-net with Adam lr=0.001", 
#              "Residual U-net with RMSprop lr=0.001", 
#              #"Residual U-net with Adam 0.0001, 0.5",
#              #"Residual U-net with Adam lr=0.0008",
#              #"Residual U-net with Adam lr=0.001 2",
#              #"Residual U-net with Adam lr=0.0004 2",
#              #"RMSprop lr = 0.0001"
#              #"Residual U-net with Adam lr=0.0004 add conv", 
#              #"Residual U-net with Adam lr=0.0004 3"
#              ]
#plot_compare(paths, path_names)
#
#### 1:10k for optimizer
#paths = ["2018-07-19 15-14-38_10/", 
#         "U128_2019-03-11 13-05-20_10/",
#         "U128_2019-03-11 08-27-43_10/",
#         "U128_2019-03-10 12-26-47_10/",
#         "U128_2019-03-10 14-55-12_10/",
#         "U128_2019-03-11 10-47-43_10/",
#         #"2019-02-21 16-24-04_10/",
#         ]
#
#path_names = ['U-net',
#              "Residual U-net with Adadelta",
#              "Residual U-net with Adam lr=0.0001",
#              "Residual U-net with Adam lr=0.0004",
#              "Residual U-net with Adam lr=0.001",
#              "Residual U-net with RMSprop lr=0.001",
#              #"Residual Unet with L2 Norm 0.004-0.5",
#              ]
#plot_compare(paths, path_names)


###################### GAN ######################

is_plot_imagegan = True
is_plot_patchgan = True

# GAN Plot 25k
#"U128GAN_2019-03-13 17-10-15_25/","PatchGAN_p2p_128_batch4_mse_mse_1_100", # Worst ever
#"PatchGAN mse:mae $\lambda=$ 100 x", "U128GAN_2019-03-14 14-23-22_25/",

paths = ["2018-07-20 07-53-50_25/", "2019-02-28 11-05-31_25/"]
         
imagegan = ["U128GAN_2019-03-17 08-30-55_25/", # Scheduled, Not yet finished
            "U128GAN_2019-03-17 08-40-07_25/", # Scheduled, Not yet finished
            "U128GAN_2019-03-18 19-28-40_25/",
            #"U128GAN_2019-03-18 19-28-32_25/",
            ]

## Old calculated         
#patchgan = ["U128GAN_2019-03-16 20-39-33_25/",
#            "GAN_2019-03-04 20-08-40_25/", 
#            "GAN_2019-03-05 11-08-37_25/", 
#            "GAN_2019-03-05 15-34-13_25/",
#            ]

# New calculated
patchgan = ["U128GAN_2019-03-16 20-39-33_25/",
            "U128GAN_2019-03-19 15-18-43_25/", #"U128GAN_2019-03-19 08-45-54_25/",
            "U128GAN_2019-03-19 18-36-56_25/", #"U128GAN_2019-03-19 10-33-50_25/",
            #"U128GAN_2019-03-19 13-04-25_25/",
            ]
     
 #"U128GAN_2019-03-12 17-10-14_25/", 
 #"U128GAN_2019-03-14 14-24-43_25/",
 #"GAN_2019-03-07 08-43-46_25/",
 #"U256_2019-03-09 19-21-38_25/", 
 #"U128GAN_2019-03-12 21-16-58_25/",
 #"U128GAN_2019-03-14 12-30-16_25/",
 
 # cross + mse make worse
 #"U128GAN_2019-03-13 14-21-56_25/", "U128GAN_2019-03-13 17-12-47_25/",
 #"U128GAN_2019-03-14 10-33-47_25/", "U128GAN_2019-03-14 09-02-14_25/"
 #"GAN_2019-03-06 08-59-42_25/","GAN_2019-03-06 14-18-58_25/", 
         

path_names = ['U-net', "Residual U-net"]
              
imagegan_names = [r"ImageGAN $\lambda=1000$", #DCGAN 15-128_batch16_mse_mae
                  r"ImageGAN $\lambda=100$", #DCGAN 15-128_batch16_mse_mae
                  r"ImageGAN $\lambda=10$", #DCGAN 15-128_batch16_mse_mae
                  #r"ImageGAN $\lambda=1$", #DCGAN 15-128_batch16_mse_mae
                  ]
              
patchgan_names = [r"PatchGAN $\lambda=1000$",  # mse:mae 
                  r"PatchGAN $\lambda=100$",  # mse:mae 
                  r"PatchGAN $\lambda=10$",   # mse:mae 
                  #r"PatchGAN $\lambda=1$",    # mse:mae 
                  ]

path_names = path_names + imagegan_names + patchgan_names

  #"ImageGAN DCGAN crossentropy_mse_1_100_ 25-128_batch16", 
  #"ImageGAN DCGAN crossentropy_mse_1_10_ 25-128_batch16", 
  #r"PatchGAN 'mse', 'mse', [1, 100]",
  #"256 at 25k", 
  #"SRGAN 25-128",
  #"PatchGAN_p2p_128_batch16_mse_mae_1_100",
  
  
  # cross + mse make worse
  #"PatchGAN_p2p_128_batch4_crossentropy_mse_1_100", "PatchGAN_p2p_128_batch4_crossentropy_mse_1_10",
  #"PatchGAN_p2p_128_batch16_crossentropy_mse_1_100", "PatchGAN_p2p_128_batch4_crossentropy_mse_1_10 x",
  #r"GAN PatchGAN mse:mae $\lambda=$ 1:1000",r"GAN PatcheGAN mse:mae $\lambda=$ 10:1", 
          
if is_plot_imagegan:
    paths = paths + imagegan 
    path_names = path_names + imagegan_names
if is_plot_patchgan:
    paths = paths + patchgan    
    path_names = path_names + patchgan_names        
plot_compare(paths, path_names)


## GAN Plot 15k
paths = ["2018-07-19 13-13-22_15/", "2019-02-26 12-55-00_15/"]
         
imagegan = ["U128GAN_2019-03-15 21-30-01_15/",
            "U128GAN_2019-03-19 18-33-51_15/", #"U128GAN_2019-03-15 13-10-22_15/",
            "U128GAN_2019-03-15 09-07-17_15/",
            #"U128GAN_2019-03-15 15-22-19_15/",
            ]
         
patchgan = ["U128GAN_2019-03-15 12-48-30_15/",
            "U128GAN_2019-03-14 16-46-17_15/",
            "U128GAN_2019-03-15 08-45-34_15/",
            #"U128GAN_2019-03-15 10-55-05_15/",
            ]
         
#"GAN_2019-03-07 20-17-46_15/",
#"GAN_2019-03-08 13-11-47_15/",
#"U128GAN_2019-03-12 12-30-22_15/",
#"U128GAN_2019-03-14 16-47-10_15/",

path_names = ['U-net', "Residual U-net"] # Current best
              
imagegan_names = [r"ImageGAN $\lambda=1000$", #DCGAN 15-128_batch16_mse_mae
                  r"ImageGAN $\lambda=100$", #DCGAN 15-128_batch16_mse_mae
                  r"ImageGAN $\lambda=10$", #DCGAN 15-128_batch16_mse_mae
                  #r"ImageGAN $\lambda=1$", #DCGAN 15-128_batch16_mse_mae
                  ]

patchgan_names = [r"PatchGAN $\lambda=1000$",  # 15-128_batch16_mse_mae
                  r"PatchGAN $\lambda=100$",    # 15-128_batch16_mse_mae
                  r"PatchGAN $\lambda=10$",     # 15-128_batch16_mse_mae
                  #r"PatchGAN $\lambda=1$",      # 15-128_batch16_mse_mae
                  ]

#"GAN Patch",
#"GAN Simple",
#"DCGAN 15-128",
#"DCGAN 15-128_batch16_crossentropy_mse_1_10",

if is_plot_imagegan:
    paths = paths + imagegan 
    path_names = path_names + imagegan_names
if is_plot_patchgan:
    paths = paths + patchgan    
    path_names = path_names + patchgan_names                   
plot_compare(paths, path_names)


## GAN Plot 10k
paths = ["2018-07-19 15-14-38_10/", "U128_2019-03-10 12-26-47_10/"]
         
imagegan = ["U128GAN_2019-03-16 19-36-21_10/",
            "U128GAN_2019-03-16 10-41-20_10/",
            "U128GAN_2019-03-16 14-37-49_10/",
            #"U128GAN_2019-03-19 08-43-25_10/"
            ]
         
patchgan = ["U128GAN_2019-03-15 14-56-54_10/",
            "U128GAN_2019-03-16 10-38-41_10/",
            "U128GAN_2019-03-16 13-54-40_10/",
            #"U128GAN_2019-03-16 16-47-42_10/"
            ]

path_names = ['U-net', "Residual U-net"]
              
imagegan_names = [r"ImageGAN $\lambda=1000$", #DCGAN 15-128_batch16_mse_mae
                  r"ImageGAN $\lambda=100$", #DCGAN 15-128_batch16_mse_mae
                  r"ImageGAN $\lambda=10$", #DCGAN 15-128_batch16_mse_mae
                  #r"ImageGAN $\lambda=1$", #DCGAN 15-128_batch16_mse_mae
                  ]

patchgan_names = [r"PatchGAN $\lambda=1000$",  # 15-128_batch16_mse_mae
                  r"PatchGAN $\lambda=100$",  # 15-128_batch16_mse_mae
                  r"PatchGAN $\lambda=10$",
                  #r"PatchGAN $\lambda=1$",
                  ]

if is_plot_imagegan:
    paths = paths + imagegan 
    path_names = path_names + imagegan_names
if is_plot_patchgan:
    paths = paths + patchgan    
    path_names = path_names + patchgan_names  
plot_compare(paths, path_names)

###################### 256 x 256 ######################

#paths = ["2019-03-09 13-18-43_15/", "U256_2019-03-09 19-21-38_25/"]
#path_names = ["256 at 15k", "256 at 25k"]





