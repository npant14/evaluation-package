"""
runs an evaluation based on the Hangul characters dataset
"""
from utils import FeatureExtractor
import torch
torch.manual_seed(1)
import numpy as np
np.random.seed(1)
from torchvision import transforms,datasets
import csv
import random
import statistics


class Invert:
    def __call__(self, sample):
        inverted_image = (-1 * sample) + 1
        return inverted_image

class korean_dataloader():
    def __init__(self, image_size, datadir):
        super().__init__()

        # Directory to load Data
        self.datadir = datadir
        self.image_size = int(image_size)
        self.data = datasets.ImageFolder(root=
            self.datadir,
            transform=
            transforms.Compose([
                transforms.ToTensor(),
                Invert(),
                transforms.ToPILImage(),
                transforms.Resize((40,40)),
                transforms.Pad(((self.image_size - 40)//2, (self.image_size - 40)//2)),
                transforms.ToTensor(),
            ]))

    def __getitem__(self, idx):
        img = self.data[idx]
        return img

    def __len__(self):
        return len(self.data)


class Korean():
    def __init__(self, model, outdir, device, data_dir, img_size=322):
        self.model = model
        self.outdir = outdir
        self.device = device
        self.data_dir = data_dir
        self.img_size = img_size
        self.data = korean_dataloader(img_size, data_dir)
        print("setup Korean experiment -- ready to run")

    '''
    This function is expecting a square image that has a character 
    that is 40x40 pixels in the center of the image

    1 < size < image.shape[1]
    size is the new size of the character
    '''
    def resize_image(self, image, size):
        image_size = image.shape[1]
        pil_image = transforms.ToPILImage()(image)
        pad_before = (image_size - size)//2
        pad_after = (image_size - size)//2
        if pad_after + pad_before + size != image_size:
            pad_after += (image_size - (pad_after + pad_before + size))
        resize = transforms.Compose([
                            transforms.CenterCrop(40),
                            transforms.Resize((size,size)),
                            transforms.Pad((pad_before, pad_before, pad_after, pad_after)),
                            transforms.ToTensor()
                        ])
        
        new_img = resize(pil_image)

        return new_img

    '''
    returns the pearson correlation for a pair of tensors
    '''
    def get_pearson_correlation(self, tensor_1, tensor_2):
        tensor_features = torch.stack((tensor_1, tensor_2))
        return torch.corrcoef(tensor_features)[0][1].item()

    '''
    parameters:
    pairs -- the pairs of sizes to calculate correlations for 
            expects list of tuples (ex [(20,20), (20,80), (20,200)])
    '''
    def create_correlation_matrices(self, pairs, layer_name):

        activations_dict = {}
        correlation_matix = np.zeros((len(self.data), len(self.data)))

        for size_1, size_2 in pairs:
            print(f"running {size_1}, {size_2}")
            for i in range(len(self.data)):
                for j in range(len(self.data)):
                    layer_features = FeatureExtractor(self.model, [layer_name])
                    resized_img = self.resize_image(self.data[i][0], size_1)
                    features = layer_features(torch.unsqueeze(resized_img, 0).to(self.device))
                    tensor_feature = features[layer_name][0][0]
                    rowfeat = torch.squeeze(torch.flatten(tensor_feature))


                    layer_features = FeatureExtractor(self.model, [layer_name])
                    resized_img = self.resize_image(self.data[j][0], size_2)
                    features = layer_features(torch.unsqueeze(resized_img, 0).to(self.device))
                    tensor_feature = features[layer_name][0][0]
                    colfeat = torch.squeeze(torch.flatten(tensor_feature))

                    ij_corr = self.get_pearson_correlation(rowfeat, colfeat)
                    correlation_matix[i][j] = ij_corr
                    del(layer_features)
                    del(features)
                    del(rowfeat)
                    del(colfeat)


            with open(f'{self.outdir}korean/{size_1}-{size_2}.csv', 'w') as f:
                print(f"writing out {size_1} {size_2} to csv")
                writer = csv.writer(f)
                for row in correlation_matix:
                    writer.writerow(row)


    def get_accuracy(self, filepaths):
        means = {}
        errs = {}
        maxes = {}
        # iterate through all the saved csvs
        for path in filepaths:
            target_size, test_size = path.split("/")[-1].split(".")[0].split("-")
            # want to check both directions
            for transpose in [False, True]:
                CSVData = open(path)
                correlations = np.loadtxt(CSVData, delimiter=",")

                if transpose:
                    correlations = correlations.transpose()
                    target_size, test_size = test_size, target_size

                normalized = correlations
                normalized = (correlations - np.min(correlations, axis=0)) / (np.max(correlations) - np.min(correlations))

                all_correct = []
                all_distractor = []
                for i in range(0, 53, 2):
                    all_correct.append(normalized[i][i])
                    all_distractor.append(normalized[i][i + 1])

                collect = []

                for _ in range(1000):
                    randidxs = random.sample(range(54), k=41)
                    randidxs.sort()

                    thresh_correct = []
                    test_correct = []
                    thresh_distractor = []
                    test_distractor = []
                    idx = 0

                    for idx, val in enumerate(all_correct):
                        if idx in randidxs:
                            thresh_correct.append(val)
                        else:
                            test_correct.append(val)

                    for idx, val in enumerate(all_distractor):
                        if idx+27 in randidxs:
                            thresh_distractor.append(val)
                        else:
                            test_distractor.append(val)

                    correct = thresh_correct
                    distractor = thresh_distractor


                    best_threshold = 0
                    best_accuracy = 0

                    for thresh in correct + distractor:
                        correctly_above_threshold = sum(i > thresh for i in correct)
                        incorrectly_above_threshold = sum(i > thresh for i in distractor)
                        correctly_below_threshold = (27) - incorrectly_above_threshold
                        acc = (correctly_above_threshold + correctly_below_threshold)/(54)
                        if acc >= best_accuracy:
                            best_accuracy = acc
                            best_threshold = thresh

                    test_correctly_above_threshold = sum(i > best_threshold for i in correct + test_correct)
                    test_incorrectly_above_threshold = sum(i > best_threshold for i in distractor + test_distractor)
                    test_correctly_below_threshold = (27) - test_incorrectly_above_threshold
                    test_acc = (test_correctly_above_threshold + test_correctly_below_threshold)/(54)

                    collect.append(test_acc)

                print(f"average test accuracy : {sum(collect)/len(collect)}")
                means[(target_size, test_size)] = sum(collect)/len(collect)
                print(f"max test accuracy : {max(collect)}")
                maxes[(target_size, test_size)] = max(collect)
                print(f"std test accuracy : {statistics.pstdev(collect)}")
                errs[(target_size, test_size)] = statistics.pstdev(collect)
                
        return means

    def run(self):
        pairs = [(13, 13), (13, 52), (13, 130)]
        layer_name = "s3.layer.3.conv3"
        filepaths = [f"{self.outdir}/korean/{size_1}-{size_2}.csv" for (size_1, size_2) in pairs]
        # self.create_correlation_matrices(pairs, layer_name)
        accs = self.get_accuracy(filepaths)
        print(accs)
        return accs