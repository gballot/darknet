#ifndef FSPT_LAYER_H
#define FSPT_LAYER_H

#include "darknet.h"

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
 * \param min_samples The minimum number of training + empty samples per node.
 * \param max_depth The maximum depth of the tree.
 * \param batch The number of images per batch.
 * \param activation The activation function applied to the input features
 *                   before storing them as input for fspts. Useful to restrict
 *                   the inputs to feature_limit.
 * \return The newly created fspt_layer.
 */
extern layer make_fspt_layer(int inputs, int *input_layers,
        int yolo_layer, network *net, float yolo_thresh,
        float *feature_limit, float *feature_importance,
        criterion_func criterion, score_func score, int min_samples,
        int max_depth, int batch, ACTIVATION activation);

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
 * \param dets Output parameter. Will be filled with the detections.
 *             Make sure to allocate enought space.
 * \return The number of detections.
 */
extern int get_fspt_detections(layer l, int w, int h, network *net,
        float yolo_thresh, float fspt_thresh, int *map, int relative,
        detection *dets);

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

#endif /* FSPT_LAYER_H */
