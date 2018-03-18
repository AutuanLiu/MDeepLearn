# fastai 库的基本结构

```
Email: autuanliu@163.com
Date: 2018/3/18
```

## 基本模块

1.  models
    1. cifar10: [kuangliu/pytorch-cifar: 95.16% on CIFAR10 with PyTorch](https://github.com/kuangliu/pytorch-cifar)
        1. main_dxy
        2. main_kuangliu
        3. preact_resnet
            ```python
            def PreActResNet18(): return PreActResNet(PreActBlock, [2,2,2,2])
                """Pre-activation ResNet in PyTorch. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, Identity Mappings in Deep Residual Networks. arXiv:1603.05027
                """

            def PreActResNet34(): return PreActResNet(PreActBlock, [3,4,6,3])
            def PreActResNet50(): return PreActResNet(PreActBottleneck, [3,4,6,3])
            def PreActResNet101(): return PreActResNet(PreActBottleneck, [3,4,23,3])
            def PreActResNet152(): return PreActResNet(PreActBottleneck, [3,8,36,3])
            ```
        4. resnext
            ```python
            def resnext29_16_64(num_classes=10):
                """Constructs a ResNeXt-29, 16*64d model for CIFAR-10 (by default)
                
                Args:
                    num_classes (uint): number of classes
                """
                model = CifarResNeXt(ResNeXtBottleneck, 29, 16, 64, num_classes)
                return model

                def resnext29_8_64(num_classes=10):
                """Constructs a ResNeXt-29, 8*64d model for CIFAR-10 (by default)
                
                Args:
                    num_classes (uint): number of classes
                """
                model = CifarResNeXt(ResNeXtBottleneck, 29, 8, 64, num_classes)
                return model
            ```
        5. senet
            ```python
            def SENet18(): 
                """SENet is the winner of ImageNet-2017 (https://arxiv.org/abs/1709.01507)."""
                return SENet(PreActBlock, [2,2,2,2])

            def SENet34(): 
                return SENet(PreActBlock, [3,4,6,3])
            ```
        6. utils_kuangliu
            ```python
            def get_mean_and_std(dataset):
                '''Compute the mean and std value of dataset.'''
            ```

            ```python
            def init_params(net):
                '''Init layer parameters.'''
            ```

            ```python
            def progress_bar(current, total, msg=None): 
                '''progress bar mimic xlua.progress.'''
            ```
        7. utils
            ```python
            class AverageMeter(object):
                """Computes and stores the average and current value"""
            ```

            ```python
            class RecorderMeter(object):
                """Computes and stores the minimum loss value and its epoch index"""
            ```
    2. convert_torch
        ```python
        def torch_to_pytorch(t7_filename,outputname=None):
        ```
    3. inceptionresnetv2
        ```python
        def inceptionresnetv2(pretrained=True):
            r"""InceptionResnetV2 model architecture from the
            `"InceptionV4, Inception-ResNet..." <https://arxiv.org/abs/1602.07261>`_ paper.

            Args:
                pretrained ('string'): If True, returns a model pre-trained on ImageNet
            """
        ```
    4. inceptionv4
        ```python
        def inceptionv4(pretrained=True):
        ```
    5. nasnet
        ```python
        def nasnetalarge(num_classes=1000, pretrained='imagenet'):
            r"""NASNetALarge model architecture from the
            `"NASNet" <https://arxiv.org/abs/1707.07012>`_ paper.
            """
        ```
    6. resnext_50_32x4d
        ```python
        def resnext_50_32x4d()
        ```
    7. resnext_101_32x4d
        ```python
        def resnext_101_32x4d()
        ```
    8. resnext_101_64x4d
        ```python
        def resnext_101_64x4d()
        ```
    9. wrn_50_2f
        ```python
        def wrn_50_2f()
        ```
2.  adaptive_softmax
    ```python
    class AdaptiveSoftmax(nn.Module)
    class AdaptiveLoss(nn.Module)
    ```
3.  column_data
    ```python
    class PassthruDataset(Dataset)
    class ColumnarDataset(Dataset)
    class ColumnarModelData(ModelData)
    class MixedInputModel(nn.Module)
    class StructuredLearner(Learner)
    class StructuredModel(BasicModel)
    class CollabFilterDataset(Dataset)
    class EmbeddingDotBias(nn.Module)
    class CollabFilterLearner(Learner)
    class CollabFilterModel(BasicModel)
    ```
4.  conv_learner
    ```python
    class ConvLearner(Learner)
    ```
5.  core
    ```python
    def sum_geom(a,r,n): 
        return a*n if r==1 else math.ceil(a*(1-r**n)/(1-r))
    def A(*a):
        return np.array(a[0]) if len(a)==1 else [np.array(o) for o in a]
    def T(a)
    ```

    ```python
    def create_variable(x, volatile, requires_grad=False):
        if not isinstance(x, Variable):
            x = Variable(T(x), volatile=volatile, requires_grad=requires_grad)
        return x
    ```

    ```python
    def V_(x, requires_grad=False)
    def V(x, requires_grad=False)
    def VV_(x)
    def VV(x)
    def to_np(v)
    def to_gpu(x, *args, **kwargs)
    def noop(*args, **kwargs)
    def split_by_idxs(seq, idxs)
    def trainable_params_(m)
    def chain_params(p)
    def set_trainable_attr(m,b)
    def apply_leaf(m, f)
    def set_trainable(l, b)
    def SGD_Momentum(momentum)
    def one_hot(a,c): return np.eye(c)[a]
    def partition(a, sz)
    def partition_by_cores(a)
    def num_cpus()
    class BasicModel()
    class SingleModel(BasicModel)
    class SimpleNet(nn.Module)
    def save(fn, a)
    def load(fn)
    def load2(fn)
    def load_array(fname)
    ```
6.  dataloader
    ```python
    def get_tensor(batch, pin)
    class DataLoader(object)
    ```
7.  dataset
    ```python
    def get_cv_idxs(n, cv_idx=0, val_pct=0.2, seed=42)
    def resize_img(fname, targ, path, new_path)
    def resize_imgs(fnames, targ, path, new_path)
    def read_dir(path, folder)
    def read_dirs(path, folder)
    def create_sample(path, r):
        """ Takes a path to a dataset and creates a sample of specified size at <path>_sample

        Parameters:
        -----------
        path: dataset path
        r (float): proportion of examples to use as sample, in the range from 0 to 1
        """
    def create_val(path, r):
        """ Takes a path to a dataset and creates a validation set of specified size

        Note - this changes the dataset at <path> by moving files to the val set

        Parameters:
        -----------
        path: dataset path
        r (float): proportion of examples to use for validation, in the range from 0 to 1

        """
    def copy_or_move_with_subdirs(subdir_lst, src, dst, r, move=False)
    def n_hot(ids, c)
    def folder_source(path, folder)
    def parse_csv_labels(fn, skip_header=True):
        """Parse filenames and label sets from a CSV file.

        This method expects that the csv file at path :fn: has two columns. If it
        has a header, :skip_header: should be set to True. The labels in the
        label set are expected to be space separated.

        Arguments:
            fn: Path to a CSV file.
            skip_header: A boolean flag indicating whether to skip the header.

        Returns:
            a four-tuple of (
                sorted image filenames,
                a dictionary of filenames and corresponding labels,
                a sorted set of unique labels,
                a dictionary of labels to their corresponding index, which will
                be one-hot encoded.)
        """
    def nhot_labels(label2idx, csv_labels, fnames, c)
    def csv_source(folder, csv_file, skip_header=True, suffix='', continuous=False)
    def dict_source(folder, fnames, csv_labels, suffix='', continuous=False)
    class BaseDataset(Dataset)
    def open_image(fn)
    class FilesDataset(BaseDataset)
    class FilesArrayDataset(FilesDataset)
    class FilesIndexArrayDataset(FilesArrayDataset)
    class FilesNhotArrayDataset(FilesArrayDataset)
    class FilesIndexArrayRegressionDataset(FilesArrayDataset)
    class ArraysDataset(BaseDataset)
    class ArraysIndexDataset(ArraysDataset)
    class ArraysNhotDataset(ArraysDataset)
    class ModelData()
    class ImageData(ModelData)
    class ImageClassifierData(ImageData)
    ```
8.  imports
9.  initializers
10. io
    ```python
    class TqdmUpTo(tqdm)
    def get_data(url, filename)
    ```
11. layer_optimizer
    ```python
    def opt_params(parm, lr, wd)
    class LayerOptimizer()
    ```
12. layers
    ```python
    class AdaptiveConcatPool2d(nn.Module)
    class Lambda(nn.Module)
    class Flatten(nn.Module)
    ```
13. learner
    ```python
    class Learner():
        def bn_freeze(self, do_freeze)
        def freeze_to(self, n)
        def unfreeze(self)
        def save(self, name)
        def load(self, name)
        def get_model_path(self, name)
        def fit(self, lrs, n_cycle, wds=None, **kwargs):

            """Method gets an instance of LayerOptimizer and delegates to self.fit_gen(..)

            Note that one can specify a list of learning rates which, when appropriately
            defined, will be applied to different segments of an architecture. This seems
            mostly relevant to ImageNet-trained models, where we want to alter the layers
            closest to the images by much smaller amounts.

            Likewise, a single or list of weight decay parameters can be specified, which
            if appropriate for a model, will apply variable weight decay parameters to
            different segments of the model.

            Args:
                lrs (float or list(float)): learning rate for the model

                n_cycle (int): number of cycles (or iterations) to fit the model for

                wds (float or list(float)): weight decay parameter(s).

                kwargs: other arguments

            Returns:
                None
            """
        def lr_find(self, start_lr=1e-5, end_lr=10, wds=None, linear=False):
            """Helps you find an optimal learning rate for a model.

            It uses the technique developed in the 2015 paper
            `Cyclical Learning Rates for Training Neural Networks`, where
            we simply keep increasing the learning rate from a very small value,
            until the loss starts decreasing.

            Args:
                start_lr (float/numpy array) : Passing in a numpy array allows you
                    to specify learning rates for a learner's layer_groups
                end_lr (float) : The maximum learning rate to try.
                wds (iterable/float)

            Examples:
                As training moves us closer to the optimal weights for a model,
                the optimal learning rate will be smaller. We can take advantage of
                that knowledge and provide lr_find() with a starting learning rate
                1000x smaller than the model's current learning rate as such:

                >> learn.lr_find(lr/1000)

                >> lrs = np.array([ 1e-4, 1e-3, 1e-2 ])
                >> learn.lr_find(lrs / 1000)

            Notes:
                lr_find() may finish before going through each batch of examples if
                the loss decreases enough.

            .. _Cyclical Learning Rates for Training Neural Networks:
                http://arxiv.org/abs/1506.01186

            """
        def predict(self, is_test=False)
        def TTA(self, n_aug=4, is_test=False):
            """ Predict with Test Time Augmentation (TTA)

            Additional to the original test/validation images, apply image augmentation to them
            (just like for training images) and calculate the mean of predictions. The intent
            is to increase the accuracy of predictions by examining the images using multiple
            perspectives.

            Args:
                n_aug: a number of augmentation images to use per original image
                is_test: indicate to use test images; otherwise use validation images

            Returns:
                (tuple): a tuple containing:

                    log predictions (numpy.ndarray): log predictions (i.e. `np.exp(log_preds)` will return probabilities)
                    targs (numpy.ndarray): target values when `is_test==False`; zeros otherwise.
            """
    ```
14. lm_rnn
    ```python
    def seq2seq_reg(output, xtra, loss, alpha=0, beta=0)
    class RNN_Encoder(nn.Module)
    class MultiBatchRNN(RNN_Encoder)
    class LinearDecoder(nn.Module)
    class LinearBlock(nn.Module)
    class PoolingLinearClassifier(nn.Module)
    class SequentialRNN(nn.Sequential)
    def get_language_model(n_tok, emb_sz, nhid, nlayers, pad_token,
                 dropout=0.4, dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5, tie_weights=True):
        """Returns a SequentialRNN model.

        A RNN_Encoder layer is instantiated using the parameters provided.

        This is followed by the creation of a LinearDecoder layer.

        Also by default (i.e. tie_weights = True), the embedding matrix used in the RNN_Encoder
        is used to  instantiate the weights for the LinearDecoder layer.

        The SequentialRNN layer is the native torch's Sequential wrapper that puts the RNN_Encoder and
        LinearDecoder layers sequentially in the model.

        Args:
            ntoken (int): number of vocabulary (or tokens) in the source dataset
            emb_sz (int): the embedding size to use to encode each token
            nhid (int): number of hidden activation per LSTM layer
            nlayers (int): number of LSTM layers to use in the architecture
            pad_token (int): the int value used for padding text.
            dropouth (float): dropout to apply to the activations going from one LSTM layer to another
            dropouti (float): dropout to apply to the input layer.
            dropoute (float): dropout to apply to the embedding layer.
            wdrop (float): dropout used for a LSTM's internal (or hidden) recurrent weights.
            tie_weights (bool): decide if the weights of the embedding matrix in the RNN encoder should be tied to the
                weights of the LinearDecoder layer.
        Returns:
            A SequentialRNN model
        """
    def get_rnn_classifer(bptt, max_seq, n_class, n_tok, emb_sz, n_hid, n_layers, pad_token, layers, drops, bidir=False,
                      dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5)
    ```
15. losses
    ```python
    def fbeta_torch(y_true, y_pred, beta, threshold, eps=1e-9)
    ```
16. metrics
    ```python
    def accuracy_np(preds, targs)
    def accuracy(preds, targs)
    def accuracy_thresh(thresh)
    def accuracy_multi(preds, targs, thresh)
    def accuracy_multi_np(preds, targs, thresh)
    ```
17. model
    ```python
    def cut_model(m, cut)
    def predict_to_bcolz(m, gen, arr, workers=4)
    def num_features(m)
    class Stepper():
        def reset(self, train=True)
        def step(self, xs, y, epoch)
        def evaluate(self, xs, y)
    def set_train_mode(m)
    def fit(model, data, epochs, opt, crit, metrics=None, callbacks=None, stepper=Stepper, **kwargs):
        """ Fits a model

        Arguments:
        model (model): any pytorch module
            net = to_gpu(net)
        data (ModelData): see ModelData class and subclasses
        opt: optimizer. Example: opt=optim.Adam(net.parameters())
        epochs(int): number of epochs
        crit: loss function to optimize. Example: F.cross_entropy
        """
    def validate(stepper, dl, metrics)
    def get_prediction(x)
    def predict(m, dl)
    def predict_with_targs(m, dl)
    def model_summary(m, input_size)
    ```
18. nlp
    ```python
    def calc_pr(y_i, x, y, b)
    def calc_r(y_i, x, y)
    def flip_tensor(x, dim)
    class RNN_Learner(Learner)
    class ConcatTextDataset(torchtext.data.Dataset)
    class LanguageModelData()
    class TextDataLoader()
    ```
19. plots
    ```python
    def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None, maintitle=None
    def plots_from_files(imspaths, figsize=(10,5), rows=1, titles=None, maintitle=None):
        """Plots images given image files.
        
        Arguments:
            im_paths (list): list of paths
            figsize (tuple): figure size
            rows (int): number of rows
            titles (list): list of titles
            maintitle (string): main title
        """
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, figsize=None):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        (This function is copied from the scikit docs.)
        """
    def plots_raw(ims, figsize=(12,6), rows=1, titles=None)
    def load_img_id(ds, idx, path)
    class ImageModelResults():
        """ Visualize the results of an image model
        
        Arguments:
            ds (dataset): a dataset which contains the images
            log_preds (numpy.ndarray): predictions for the dataset in log scale
            
        Returns:
            ImageModelResults
        """
        def plot_val_with_title(self, idxs, y):
            """ Displays the images and their probabilities of belonging to a certain class
                
                Arguments:
                    idxs (numpy.ndarray): indexes of the image samples from the dataset
                    y (int): the selected class
                    
                Returns:
                    Plots the images in n rows [rows = n]
            """
        def most_by_mask(self, mask, y, mult):
            """ Extracts the first 4 most correct/incorrect indexes from the ordered list of probabilities
            
                Arguments:
                    mask (numpy.ndarray): the mask of probabilities specific to the selected class; a boolean array with shape (num_of_samples,) which contains True where class==selected_class, and False everywhere else
                    y (int): the selected class
                    mult (int): sets the ordering; -1 descending, 1 ascending
                    
                Returns:
                    idxs (ndarray): An array of indexes of length 4
            """
        def most_uncertain_by_mask(self, mask, y):
            """ Extracts the first 4 most uncertain indexes from the ordered list of probabilities
                
                Arguments:
                    mask (numpy.ndarray): the mask of probabilities specific to the selected class; a boolean array with shape (num_of_samples,) which contains True where class==selected_class, and False everywhere else
                    y (int): the selected class
                
                Returns:
                    idxs (ndarray): An array of indexes of length 4
            """
        def most_by_correct(self, y, is_correct):
            """ Extracts the predicted classes which correspond to the selected class (y) and to the specific case (prediction is correct - is_true=True, prediction is wrong - is_true=False)
                
                Arguments:
                    y (int): the selected class
                    is_correct (boolean): a boolean flag (True, False) which specify the what to look for. Ex: True - most correct samples, False - most incorrect samples
                
                Returns:
                    idxs (numpy.ndarray): An array of indexes (numpy.ndarray)
            """
        def plot_by_correct(self, y, is_correct):
            """ Plots the images which correspond to the selected class (y) and to the specific case (prediction is correct - is_true=True, prediction is wrong - is_true=False)
                
                Arguments:
                    y (int): the selected class
                    is_correct (boolean): a boolean flag (True, False) which specify the what to look for. Ex: True - most correct samples, False - most incorrect samples
            """ 
        def most_by_uncertain(self, y):
            """ Extracts the predicted classes which correspond to the selected class (y) and have probabilities nearest to 1/number_of_classes (eg. 0.5 for 2 classes, 0.33 for 3 classes) for the selected class.
                
                Arguments:
                    y (int): the selected class
                
                Returns:
                    idxs (numpy.ndarray): An array of indexes (numpy.ndarray)
            """
        def plot_most_correct(self, y):
            """ Plots the images which correspond to the selected class (y) and are most correct.
                
                Arguments:
                    y (int): the selected class
            """
        def plot_most_incorrect(self, y): 
            """ Plots the images which correspond to the selected class (y) and are most incorrect.
                
                Arguments:
                    y (int): the selected class
            """
        def plot_most_uncertain(self, y):
            """ Plots the images which correspond to the selected class (y) and are most uncertain i.e have probabilities nearest to 1/number_of_classes.
                
                Arguments:
                    y (int): the selected class
            """
    ```
20. rnn_reg
    ```python
    def dropout_mask(x, sz, dropout)
    ```
21. rnn_train
22. set_spawn
23. sgdr
    ```python
    class SaveBestModel(LossRecorder):
    
        """ Save weigths of the model with
            the best accuracy during training.
            
            Args:
                model: the fastai model
                lr: indicate to use test images; otherwise use validation images
                name: the name of filename of the weights without '.h5'
            
            Usage:
                Briefly, you have your model 'learn' variable and call fit.
                >>> learn.fit(lr, 2, cycle_len=2, cycle_mult=1, best_save_name='mybestmodel')
                ....
                >>> learn.load('mybestmodel')
                
                For more details see http://forums.fast.ai/t/a-code-snippet-to-save-the-best-model-during-training/12066
    
        """
    ```
24. structured
    ```python
    def set_plot_sizes(sml, med, big)
    def parallel_trees(m, fn, n_jobs=8)
    def draw_tree(t, df, size=10, ratio=0.6, precision=0)
    def combine_date(years, months=1, days=1, weeks=None, hours=None, minutes=None,
              seconds=None, milliseconds=None, microseconds=None, nanoseconds=None)
    def get_sample(df,n):
        """ Gets a random sample of n rows from df, without replacement.

        Parameters:
        -----------
        df: A pandas data frame, that you wish to sample from.
        n: The number of rows you wish to sample.

        Returns:
        --------
        return value: A random sample of n rows of df.
        """
    def add_datepart(df, fldname, drop=True):
        """add_datepart converts a column of df from a datetime64 to many columns containing
        the information from the date. This applies changes inplace.

        Parameters:
        -----------
        df: A pandas data frame. df gain several new columns.
        fldname: A string that is the name of the date column you wish to expand.
            If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
        drop: If true then the original date column will be removed.
        """
    def is_date(x)
    def train_cats(df)
    def apply_cats(df, trn):
        """Changes any columns of strings in df into categorical variables using trn as
        a template for the category codes.

        Parameters:
        -----------
        df: A pandas dataframe. Any columns of strings will be changed to
            categorical values. The category codes are determined by trn.

        trn: A pandas dataframe. When creating a category for df, it looks up the
            what the category's code were in trn and makes those the category codes
            for df.
        """
    def fix_missing(df, col, name, na_dict):
        """ Fill missing data in a column of df with the median, and add a {name}_na column
        which specifies if the data was missing.

        Parameters:
        -----------
        df: The data frame that will be changed.

        col: The column of data to fix by filling in missing data.

        name: The name of the new filled column in df.

        na_dict: A dictionary of values to create na's of and the value to insert. If
            name is not a key of na_dict the median will fill any missing data. Also
            if name is not a key of na_dict and there is no missing data in col, then
            no {name}_na column is not created.
        """
    def numericalize(df, col, name, max_n_cat)
    def scale_vars(df, mapper)
    def proc_df(df, y_fld=None, skip_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None)
    def get_nn_mappers(df, cat_vars, contin_vars)
    def set_rf_samples(n)
    def reset_rf_samples()
    ```
25. text
    ```python
    class TextDataset(Dataset)
    class SortSampler(Sampler)
    class SortishSampler(Sampler)
    class LanguageModelLoader()
    class RNN_Learner(Learner)
    ```
26. torch_imports
    ```python
    def save_model(m, p)
    def load_model(m, p)
    def load_pre(pre, f, fn)
    ```
27. transforms
    ```python
    def scale_min(im, targ, interpolation=cv2.INTER_AREA):
        """ Scales the image so that the smallest axis is of size targ.

        Arguments:
            im (array): image
            targ (int): target size
        """
    def zoom_cv(x,z)
    def stretch_cv(x,sr,sc,interpolation=cv2.INTER_AREA)
    def dihedral(x, dih)
    def lighting(im, b, c)
    def rotate_cv(im, deg, mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_AREA):
        """ Rotates an image by deg degrees

        Arguments:
            deg (float): degree to rotate.
        """
    def no_crop(im, min_sz=None, interpolation=cv2.INTER_AREA)
    def center_crop(im, min_sz=None)
    def scale_to(x, ratio, targ)
    def crop(im, r, c, sz)
    class Normalize()
    class RandomRotateZoom()
    def to_bb(YY, y)
    def coords2px(y, x):
        """ Transforming coordinates to pixels.

        Arguments:
            y (np array): vector in which (y[0], y[1]) and (y[2], y[3]) are the
                the corners of a bounding box.
            x (image): an image
        Returns:
            Y (image): of shape x.shape
        """
    def random_px_rect(y, x)
    def compose(im, y, fns)
    def image_gen(normalizer, denorm, sz, tfms=None, max_zoom=None, pad=0, crop_type=None, tfm_y=None, pad_mode=cv2.BORDER_REFLECT)
    def tfms_from_stats(stats, sz, aug_tfms=None, max_zoom=None, pad=0, crop_type=CropType.RANDOM, tfm_y=None,
        pad_mode=cv2.BORDER_REFLECT)
    def tfms_from_model(f_model, sz, aug_tfms=None, max_zoom=None, pad=0, crop_type=CropType.RANDOM, tfm_y=None,
        pad_mode=cv2.BORDER_REFLECT)
    ```
28. utils
    ```python
    def gray(img)
    def to_plot(img)
    def plot(img)
    def floor(x)
    def ceil(x)
    def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None)
    def do_clip(arr, mx)
    def wrap_config(layer)
    def copy_layer(layer)
    def copy_layers(layers)
    def copy_weights(from_layers, to_layers)
    def save_array(fname, arr)
    def laod_array(fname)
    def get_classes(path)
    def limit_mem()
    class MixIterator(object)
    ```
