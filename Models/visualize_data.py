import os 
import cv2
import matplotlib.pyplot as plt
import argparse
def main(FOLDER_PATH):
    
    names = [x for x in os.listdir(FOLDER_PATH)]

    ims_vis = []

    for class_name in names:

        #print(class_name)

        img_path = os.path.join(FOLDER_PATH, class_name) # create a directory to classes folder

        ims = [x for x in os.listdir(img_path) if x.lower().endswith('.jpg')] # list ra toan bo files trong img_path folder, giu lai cac file co jpg la duoi

        _path = os.path.join(img_path, ims[0])

        ims_vis.append(_path)
    # Create a new figure with 4 columns and as many rows as needed
    num_rows = len(names) // 4 + 1 # tinh toan so luong hang can dung

    fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(12, 3*num_rows))

    # Loop through each class and plot the first image
    for i, filepath in enumerate(ims_vis):

        # Load the image using plt.imread() and plot it on one of the subplots
        img = cv2.imread(filepath)

        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        row_idx, col_idx = i // 4, i % 4

        axes[row_idx, col_idx].imshow(im_rgb)

        axes[row_idx, col_idx].set_title(names[i])

    # Hide the x and y axis ticks for all subplots
    for ax in axes.flatten():
        
        ax.set_xticks([])

        ax.set_yticks([])

    # Adjust the spacing between subplots and display the figure
    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder-path', type = str, default = './images', help = 'images folder path' )
    # ...
    args = parser.parse_args()
    main(args.folder_path)