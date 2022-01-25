import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import cv2


def draw_heatmap(in_path, out_path, space):
    """draw heatmap given a numpy array
    Parameters:
        in_path: path of input numpy
        out_path: path of output figure 
        space: space between lines
    Returns:
        None
    """
    a = np.load(in_path)
    a = np.squeeze(a)
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(0, a.shape[0], space))
    ax.set_yticks(np.arange(0, a.shape[1], space))
    im = ax.imshow(a)
    plt.grid(b=True, which='major', color='orange', linestyle='-')
    # ax.set_yticks(np.arange(a.shape[1]))
    fig.savefig(out_path)

def filter_component(in_path, area_threshold, out_path):
    """filter connected components no greater than a given threashold
    Parameters:
        in_path: path of input numpy
        area_threshold:
        out_path: path of output figure
    Returns:
        None
    """
    image = cv2.imread(in_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # cv2.imwrite('./binary.jpg', thresh)
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    mask = np.zeros(gray.shape, dtype = 'uint8')
    for i in range(1, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > area_threshold:
            componentMask = (labels == i).astype('uint8') * 255
            mask = cv2.bitwise_or(mask, componentMask)
    cv2.imwrite(out_path, mask)


def get_con_line(in_path, out_path):
    """draw contour line given a numpy array
    Parameters:
        in_path: path of input numpy
        out_path: path of output figure 
    Returns:
        None
    """
    z = np.load(in_path)
    z = np.squeeze(z)
    xlist = np.linspace(0, z.shape[0], z.shape[0])
    ylist = np.linspace(0, z.shape[1], z.shape[1])
    x, y = np.meshgrid(xlist, ylist)
    fig, ax = plt.subplots(1,1)
    cp = ax.contourf(x, y, z)
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('Filled Contours Plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.savefig(out_path)

if __name__ == '__main__':
    filter_component('/home/hongduo/school/connected_components/Mask24.png', 200, '/home/hongduo/school/connected_components/result.png')
    get_con_line('/home/hongduo/school/contour_line/M1_test1.npy','/home/hongduo/school/contour_line/fig.jpg')