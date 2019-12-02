#ifndef FSPT_LAYER_H
#define FSPT_LAYER_H

#include "darknet.h"
#include "fspt.h"

/**
 * Creates a fspt layer.
 * This layer must be after the input layers and the yolo layer.
 * A fspt layer creates as many fspts as the number of class the attached
 * yolo layer. During the training of the fspt_layer, its fspts are fitted
 * to the input extracted from a row in the input convolutional layer
 * corresponding to the center of the ground truth box and the corresponding
 * class.
 *
 * \param inputs The number of convolutional layers from where features are
 *               extracted i.e. size of input_layers.
 * \param input_layers The index of the convolutional layers from where the
 *                     features are extracted.
 * \param yolo_layer The index of the yolo layer That this fspt layer checks.
 * \param net The network.
 * \param yolo_thresh The threshold for the yolo layer to consider its
 *                    predictions as inputs for fspt.
 * \param feature_limit The bounds for the input features.
 *                      Size [min, max]*n_features.
 * \param feature_importance The feature importance. Size n_features.
 * \param criterion Pointer to the criterion function for training.
 * \param score Pointer to the score function for training.
 * \param batch The number of images per batch.
 * \param c_args_template Argument that will be passed to the criterion
 *                        function. Can specify the max depth of the tree or
 *                        the number of samples per leaves for instance.
 * \param s_args_template Argument that will be passed to the score
 *                        function. Can specify the euristic hyperparameter
 *                        or if the score can be computed during the fitting.
 * \param activation The activation function applied to the input features
 *                   before storing them as input for fspts. Useful to restrict
 *                   the inputs to feature_limit.
 * \return The newly created fspt_layer.
 */
extern layer make_fspt_layer(int inputs, int *input_layers,
        int yolo_layer, network *net, float yolo_thresh,
        float *feature_limit, float *feature_importance,
        criterion_func criterion, score_func score, int batch,
        criterion_args c_args_template, score_args s_args_template,
        ACTIVATION activation);

/**
 * Forward for the fspt layer.
 * If the net flag train_fspt is 0. Then this just copies the output of the
 * yolo layer to the output. If the flag is true, then the data are extracted
 * from the convolutional input layers and stored to the fspt_training_data[i]
 * where i is the class of the boxes predicted by the yolo layer.
 * Note that the trees are not fitted by this function.
 *
 * \praram l The fspt layer.
 * \param net The network.
 */
extern void forward_fspt_layer(layer l, network net);

/**
 * GPU version of forward_fspt_layer.
 * If the net flag train_fspt is 0. Then this just copies the gpu output of the
 * yolo layer to the gpu output. If the flag is true, then the data are
 * extracted from the convolutional input layers and stored to the
 * fspt_training_data[i] where i is the class of the boxes predicted by the
 * yolo layer.
 * Note that the trees are not fitted by this function.
 *
 * \praram l The fspt layer.
 * \param net The network.
 */
extern void forward_fspt_layer_gpu(layer l, network net);

/**
 * Resize the layer to size w*h.
 *
 * \param l The fspt layer.
 * \param w The width in pixels.
 * \param h The height in pixels.
 */
extern void resize_fspt_layer(layer *l, int w, int h);

/**
 * Gets the detection of the yolo layer corrected by the fspts.
 *
 * \param l the fspt layer.
 * \param w The width in pixels.
 * \param h The height in pixels.
 * \param net The network.
 * \param yolo_thresh The threshold for yolo detection.
 * \param fspt_thresh The threshold for fspt rejection.
 * \param map ?
 * \param relative ?
 * \param suppress If true, the yolo detections that does not excceed the
 *                 fspt threshold are not part of the output.
 * \param dets Size batches. Output parameter. Will be filled with the
 *             detections for each images of the batch. Make sure to allocate
 *             enought space.
 * \return The number of detections for each images of the batch.
 */
extern int *get_fspt_detections_batch(layer l, int w, int h, network *net,
        float yolo_thresh, float fspt_thresh, int *map, int relative,
        int suppress, detection **dets);


/**
 * Gives the score of the fspt according to the real classes and real boxes.
 * The result is given under the form of predictions where the probability for
 * the true class is the score given by the fspt of this class and boxe size,
 * if it match to this layer mask.
 *
 * \param l The fspt layer.
 * \param net The network.
 * \param dets Output parameter of size net->batch. will be filled by the
 *             predictions for each batch image. Must be allocated by the
 *             caller. @see make_network_truth_boxes_batch.
 * \param n_boxes Output parameter of size net->batch. Will be filled by
 *                the number of detection by batch image.
 */
extern void fspt_predict_truth(layer l, network net, detection **dets,
        int **n_boxes);

/**
 * Saves all the fspts to a file. Opening and closing the file is the 
 * responsibility of the caller.
 *
 * \param l The fspt layer.
 * \param fp The file pointer.
 */
extern void save_fspt_trees(layer l, FILE *fp);

/**
 * Load all the fspts from a file. Opening and closing the file is the 
 * responsibility of the caller.
 *
 * \param l The fspt layer.
 * \param fp The file pointer.
 */
extern void load_fspt_trees(layer l, FILE *fp);

/**
 * Sets the training data for the class but don't fit.
 *
 * \param l The fspt layer.
 * \param class The class.
 * \param refit If true, sets fspt samples even if it is already fitted.
 * \param merge If true, merge new data with samples already in the tree.
 */
extern void fspt_layer_set_samples_class(layer l, int class, int refit,
        int merge);

/**
 * Fits the fspt of class class of the fspt layer.
 * The data must be already extracted.
 *
 * \param l The fspt layer.
 * \param class The class to fit.
 * \param refit If true, refit fspt even if it is already fitted.
 * \param merge If true, merge new data with samples already in the tree.
 */
extern void fspt_layer_fit_class(layer l, int class, int refit, int merge);

/**
 * Compute the score of the leaves of the fspt without rebuilding it.
 *
 * \param l The fspt layer.
 * \param class The class to re score.
 */
extern void fspt_layer_rescore_class(layer l, int class);

/**
 * Sets the training data for all the class but don't fit.
 *
 * \param l The fspt layer.
 * \param refit If true, sets fspts samples even if they are already fitted.
 * \param merge If true, merge new data with samples already in the tree.
 */
extern void fspt_layer_set_samples(layer l, int refit, int merge);

/**
 * Fits the fspts of the fspt layer.
 * The data must be already extracted.
 *
 * \param l The fspt layer.
 * \param refit If true, refit fspts even if they are already fitted.
 * \param merge If true, merge new data with samples already in the tree.
 */
extern void fspt_layer_fit(layer l, int refit, int merge);

/**
 * Rescore the fspts of the fspt layer.
 *
 * \param l The fspt layer.
 */
extern void fspt_layer_rescore(layer l);

/**
 * Merges the training data in layer l and base.
 * the training data of layer l are appended to the training data
 * of layer base.
 *
 * \param l Fspt layer source of the merge.
 * \param base Fspt layer destination of the merge.
 */
extern void merge_training_data(layer l, layer base);

#endif /* FSPT_LAYER_H */
