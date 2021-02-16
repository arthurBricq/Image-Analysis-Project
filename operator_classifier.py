# %% md
##FUNC DEF AND NEEDED IMPORTS

# %% codecell
import skimage.io
from skimage.measure import label
import skimage.morphology as morph
import matplotlib.pyplot as plt
import numpy as np

def get_nbh(img,x,y,w=3,h=3,padding=False,padding_val=0):
    ''' get neighbourhood around x,y pixel with mirrored periodisation, works
        only with odd w,h! no test if odd or even, so be careful!
        @param w,h      width of nbh, height of nbh. Must be odd
        @param x,y      position where nbh is extracted
        @param padding  instead of mirrored periodisation apllys padding at
                        boundary
    '''
    nx,ny=img.shape
    delta_w=int(w/2)
    delta_h=int(h/2)
    nbh=np.zeros((w,h))
    if padding:
        for xx in range(x-delta_w,x+delta_w+1):
            for yy in range(y-delta_h,y+delta_h+1):
                #check if values out of bounds
                if xx<0 or xx>=nx or yy<0 or yy>=ny:
                    nbh_val=padding_val
                else:
                    nbh_val=img[xx,yy]
                #put values in nbh, origin (x,y) in the middle of nbh
                nbh[xx-x+delta_w][yy-y+delta_h]=nbh_val
    return nbh

def binarize_data(img):
    #blue color r!=255
    r=img[:,:,0]
    #convert to binary
    r[r!=255]=1
    r[r!=1]=0

    return r

def count_endings(skel,foreground=1,plot_endings=False):
    ''' counts the number of endings of a skeleton based on a 8-connect mask
    '''
    nx,ny=skel.shape
    nb_endings=0
    if plot_endings:
        out=skel.copy().astype(int)

    for x in range(nx):
        for y in range(ny):
            nbh=get_nbh(skel,x,y,padding=True)
            #assuming binary nature of image, ending has only 1 neighbour
            #-> 8 connect contains only two foreground px
            if np.sum(nbh)==2*foreground and nbh[1,1]==foreground:
                nb_endings+=1
                if plot_endings:
                    out[x,y]=2

    if plot_endings:
        plt.figure()
        plt.imshow(out)

    return nb_endings

def count_elements(bin_im):
    ''' counts the number of elements of a given object
    '''
    labels=label(bin_im)

    #"-1" for background
    return np.unique(labels).size-1

def classify_object(object_features,labels={"add":10,"sub":13,"mult":14,"div":12,"eq":11, "ukwn":14}):
    ''' classification of operator object based on given features
    '''

    n_el=object_features['nb_elements']

    if n_el==3:
        return labels["div"]
    elif n_el==2:
        return labels["eq"]
    elif n_el==1:
        #add,sub,mult
        n_end=n_el=object_features['nb_endings']

        if n_end==2:
            return labels["sub"]
        elif n_end==4:
            return labels["add"]
        elif n_end==6:
            return labels["mult"]
        else:
            return labels["ukwn"]
    else:
        return labels["ukwn"]

def classify_bin_operator(bin_im,plot_ends=False):
    ''' Takes a binary image of an operator +,-*,÷,= and extracts two features
        for classification: the number of elements and the number of endings
    '''
    object_features={}

    #extract first feature
    object_features['nb_elements']=count_elements(bin_im)

    #extract second feature
    bin_im=morph.erosion(bin_im)
    skel=morph.skeletonize(bin_im)
    object_features['nb_endings']=count_endings(skel,plot_endings=plot_ends)

    #return class label
    return classify_object(object_features),object_features
