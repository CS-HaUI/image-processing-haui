import matplotlib.pyplot as plt
import cv2

class Plot:
    """
        this Plot use RGB to plot image
    """
    def __init__(self, 
                 nrows  = 1,
                 ncols  = 1,
                 images = [None],
                 figsize= (10, 10),
                 show_type = [None],
                 show_axis = [None],
                 show_title= [None],
                 init_fig  = True):
        self.nrows = nrows
        self.ncols = ncols
        self.images = images
        self.figsize = figsize
        self.show_type = show_type
        self.show_axis = show_axis
        self.show_title= show_title
        if init_fig:
            self.__init_fig()
        
    
    def __init_fig(self):
        if self.ncols == self.nrows == 1:
            if self.show_type[0] == 'GRAY':
                self.images[0] = cv2.cvtColor(self.images[0], cv2.COLOR_RGB2GRAY)
                plt.imshow(self.images[0], cmap='gray')
            elif self.show_type[0] == "RGB":
                plt.imshow(self.images[0][:, :, ::-1])
            else:
                plt.imshow(self.images[0])
            plt.axis(self.show_axis[0])
            plt.title(self.show_title[0])
            plt.show()
        else:
            fig, ax = plt.subplots(nrows= self.nrows, ncols= self.ncols, figsize= self.figsize)
            if self.nrows == 1 or self.ncols == 1:
                for index, image in enumerate(self.images):
                    if self.show_type[index] == 'GRAY' or self.show_type[index] == 'LBP':
                        ax[index].imshow(image, cmap='gray')
                    elif self.show_type[index] == "RGB":
                        ax[index].imshow(image[:, :, ::-1])
                    else:
                        ax[index].imshow(image)
                    ax[index].axis(self.show_axis[index])
                    ax[index].set_title(self.show_title[index])
                    
            else:
                for index, list in enumerate(self.images):
                    for jndex, value in enumerate(list):
                        if self.show_type[index][jndex] == 'GRAY' or self.show_type[index][jndex] == 'LBP':
                            ax[index][jndex].imshow(value, cmap='gray')
                        elif self.show_type[index][jndex] == "RGB":
                            ax[index][jndex].imshow(value[:, :, ::-1])
                        else:
                            ax[index][jndex].imshow(value)
                        ax[index][jndex].axis(self.show_axis[index][jndex])  
                        ax[index][jndex].set_title(self.show_title[index][jndex])

            
    def _plot(self):
        plt.show()
    
            
    def _save(self, file):
        plt.savefig(file)