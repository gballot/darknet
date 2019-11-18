#include "darknet.h"

#include <stdlib.h>

#include "image.h"
#include "utils.h"
#include "fspt_layer.h"
#include "network.h"

typedef struct validation_data {
    int n_images;
    int n_truth;
    int n_yolo_detections;
    int n_fspt_detections;
    int classes;
    int *n_true_detection_by_class;
    int *n_false_detection_by_class;
    int *n_no_detection_by_class;
    int *n_true_rejections_by_class;
    int *n_false_rejection_by_class;
    float iou_thresh;
} validation_data;

static void print_fspt_detections(FILE **fps, char *id, detection *dets,
        int total, int classes, int w, int h) {
    for(int i = 0; i < total; ++i) {
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(int j = 0; j < classes; ++j) {
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id,
                    dets[i].prob[j], xmin, ymin, xmax, ymax);
        }
    }
}

static void update_validation_data(int nboxes_yolo, detection *dets_yolo,
        int nboxes_fspt, detection *dets_fspt, int nboxes_truth_fspt,
        detection *dets_truth_fspt, int nboxes_truth,
        detection *dets_truth, validation_data *val) {
    //TODO
}

static void print_stats(char *datacfg, char *cfgfile, char *weightfile,
        float yolo_thresh, float fspt_thresh) {
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);

    network *net = load_network(cfgfile, weightfile, 0);

    list *fspt_layers = get_network_layers_by_type(net, FSPT);
    while (fspt_layers->size > 0) {
        layer *l = (layer *) list_pop(fspt_layers);
        for (int i = 0; i < l->classes; ++i) {
            fspt_t *fspt = l->fspts[i];
            fspt_stats *stats = get_fspt_stats(fspt, 0, NULL);
            char buf[256] = {0};
            sprintf(buf, "%s class %d", l->ref, i);
            print_fspt_stats(stderr, stats, buf);
            free_fspt_stats(stats);
        }
    }
}

void test_fspt(char *datacfg, char *cfgfile, char *weightfile, char *filename,
        float yolo_thresh, float fspt_thresh, float hier_thresh, char *outfile,
        int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net->w, net->h);
        layer l = net->layers[net->n-1];


        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input,
                what_time_is_it_now()-time);
        int nboxes_yolo = 0;
        detection *dets_yolo = get_network_boxes(net, im.w, im.h, yolo_thresh,
                hier_thresh, 0, 1, &nboxes_yolo);
        printf("%d boxes predicted by yolo.\n", nboxes_yolo);
        if (nms) do_nms_sort(dets_yolo, nboxes_yolo, l.classes, nms);
        draw_detections(im, dets_yolo, nboxes_yolo, yolo_thresh, names,
                alphabet, l.classes);
        int nboxes_fspt = 0;
        detection *dets_fspt = get_network_fspt_boxes(net, im.w, im.h,
                yolo_thresh, fspt_thresh, hier_thresh, 0, 1, &nboxes_fspt);
        printf("%d boxes predicted by fspt.\n", nboxes_fspt);
        if (nms) do_nms_sort(dets_fspt, nboxes_fspt, l.classes, nms);
        draw_fspt_detections(im, dets_fspt, nboxes_fspt, yolo_thresh, names,
                alphabet, l.classes);
        free_detections(dets_yolo, nboxes_yolo);
        free_detections(dets_fspt, nboxes_fspt);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions_fspt");
#ifdef OPENCV
            make_window("predictions_fspt", 512, 512, 0);
            show_image(im, "predictions_fspt", 0);
#endif
        }

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}

static void train_fspt(char *datacfg, char *cfgfile, char *weightfile,
        int *gpus, int ngpus, int clear, int refit, int ordered,
        int one_thread, int merge, int only_fit, int print_stats_after_fit) {
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.txt");
    char *backup_directory = option_find_str(options, "backup", "backup/");

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network **nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i) {
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;
    data train, buffer;

    layer l = net->layers[0];
    for (int i = 0; i < net->n; ++i) {
        l = net->layers[i];
        if (l.type == FSPT || l.type == YOLO)
            break;
    }
    if (l.type != FSPT && l.type != YOLO)
        error("The net must have fspt or yolo layers");

    int classes = l.classes;

    if (only_fit) { 
        fprintf(stderr, "Only fit FSPTs...\n");
    } else {
        list *plist = get_paths(train_images);
        char **paths = (char **)list_to_array(plist);

        load_args args = get_base_args(net);
        args.coords = l.coords;
        args.paths = paths;
        args.n = imgs;
        args.m = plist->size;
        args.classes = classes;
        args.jitter = l.jitter;
        args.num_boxes = l.max_boxes;
        args.d = &buffer;
        args.type = DETECTION_DATA;
        args.threads = 64;
        args.ordered = ordered;
        args.beg = 0;

        pthread_t load_thread = load_data(args);
        double time;
        if (ordered) {
            net->max_batches = plist->size / imgs;
        }
        while (get_current_batch(net) < net->max_batches) {
            time=what_time_is_it_now();
            pthread_join(load_thread, 0);
            train = buffer;
            args.beg = *net->seen;
            load_thread = load_data(args);
            printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
            time=what_time_is_it_now();
#ifdef GPU
            if(ngpus == 1){
                train_network_fspt(net, train);
            } else {
                train_networks_fspt(nets, ngpus, train, 4);
            }
#else
            train_network_fspt(net, train);
#endif
            i = get_current_batch(net);
            fprintf(stderr,
                    "%ld: %lf seconds, %d images added to fspt input\n",
                    get_current_batch(net), what_time_is_it_now()-time,
                    i*imgs);
            free_data(train);
        }
#ifdef GPU
        if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
        fspt_layers_set_samples(net, refit, merge);
        char buff[256];
        sprintf(buff, "%s/%s_data_extraction.weights", backup_directory, base);
        save_weights(net, buff);
        fprintf(stderr, "Data extraction done. Fitting FSPTs...\n");
    }
    fit_fspts(net, classes, refit, one_thread, merge);
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
    fprintf(stderr, "End of FSPT training\n");
    if (print_stats_after_fit) {
        list *fspt_layers = get_network_layers_by_type(net, FSPT);
        while (fspt_layers->size > 0) {
            layer *l = (layer *) list_pop(fspt_layers);
            for (int i = 0; i < l->classes; ++i) {
                fspt_t *fspt = l->fspts[i];
                fspt_stats *stats = get_fspt_stats(fspt, 0, NULL);
                char buf[256] = {0};
                sprintf(buf, "%s class %d", l->ref, i);
                print_fspt_stats(stderr, stats, buf);
                free_fspt_stats(stats);
            }
        }
    }
}

static void validate_fspt(char *datacfg, char *cfgfile, char *weightfile,
        float yolo_thresh, float fspt_thresh, float hier_thresh, int ngpus,
        int ordered, char *outfile) {
    //TODO
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/valid.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *base = basecfg(cfgfile);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network **nets = calloc(ngpus, sizeof(network));
    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i) {
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, 1);
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;
    data val, buffer;

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[0];
    for (int i = 0; i < net->n; ++i) {
        l = net->layers[i];
        if (l.type == FSPT || l.type == YOLO)
            break;
    }
    if (l.type != FSPT && l.type != YOLO)
        error("The net must have fspt or yolo layers");

    int classes = l.classes;

    double start = what_time_is_it_now();

    float nms = .45;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = 0;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    args.threads = 64;
    args.ordered = 1;
    args.beg = 0;

    validation_data val_data = {0};
    val_data.n_images = imgs;
    val_data.classes = classes;

    pthread_t load_thread = load_data(args);
    double time;
    if (ordered) {
        net->max_batches = plist->size / imgs;
    }
    while (get_current_batch(net) < net->max_batches) {
        time=what_time_is_it_now();
        pthread_join(load_thread, 0);
        val = buffer;
        args.beg = *net->seen;
        load_thread = load_data(args);
        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
        time=what_time_is_it_now();
#ifdef GPU
        if(ngpus == 1){
            validate_network_fspt(net, val);
        } else {
            validate_network_fspt(net, val);
            validate_networks_fspt(nets, ngpus, val, 4);
        }
#else
        validate_network_fspt(net, val);
#endif
        i = get_current_batch(net);
        fprintf(stderr,
                "%ld: %lf seconds, %d images added to validation.\n",
                get_current_batch(net), what_time_is_it_now()-time,
                i*imgs);
        // WIP
        int w = val.w;
        int h = val.h;
        /* Yolo boxes. */
        int *nboxes_yolo;
        detection **dets_yolo = get_network_boxes_batch(net, w, h, fspt_thresh,
                hier_thresh, map, 0, &nboxes_yolo);
        if (nms) {
            for (int b = 0; b < net->batch; ++b)
                do_nms_sort(dets_yolo[b], nboxes_yolo[b], classes, nms);
        }
        /* FSPT boxes. */
        int *nboxes_fspt;
        detection **dets_fspt = get_network_fspt_boxes_batch(net, w, h,
                yolo_thresh, fspt_thresh, hier_thresh, map, 0,
                &nboxes_fspt);
        if (nms) {
            for (int b = 0; b < net->batch; ++b)
                do_nms_sort(dets_fspt[b], nboxes_fspt[b], classes, nms);
        }
        /* FSPT truth boxes */
        int *nboxes_truth_fspt;
        detection **dets_truth_fspt =
            get_network_fspt_truth_boxes_batch(net, w, h,
                yolo_thresh, fspt_thresh, hier_thresh, map, 0,
                &nboxes_truth_fspt);
        /* Truth boxes */
        int *nboxes_truth;
        detection **dets_truth = get_network_truth_boxes_batch(net, w, h,
                &nboxes_truth);

        for (int b = 0; b < net->batch; ++b) {
            update_validation_data(nboxes_yolo[b], dets_yolo[b], nboxes_fspt[b],
                    dets_fspt[b], nboxes_truth_fspt[b], dets_truth_fspt[b],
                    nboxes_truth[b], dets_truth[b], &val_data);
            free_detections(dets_fspt[b], nboxes_fspt[b]);
            free_detections(dets_yolo[b], nboxes_yolo[b]);
            free_detections(dets_truth_fspt[b], nboxes_truth_fspt[b]);
            free_detections(dets_truth[b], nboxes_truth[b]);
            free(dets_fspt);
            free(dets_yolo);
            free(dets_truth_fspt);
            free(dets_truth);
            free(nboxes_fspt);
            free(nboxes_yolo);
            free(nboxes_truth_fspt);
            free(nboxes_truth);
        }
        // END WIP
        free_data(val);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    fprintf(stderr, "Total Detection Time: %f Seconds\n",
            what_time_is_it_now() - start);
}

static void validate_fspt_recall(char *cfgfile, char *weightfile) {
    //TODO
    error("TODO");
}

void run_fspt(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr,
"usage: %s %s <train/test/valid> <datacfg> <netcfg> [weights] [inputfile] [options]\n\
With :\n\
    train -> train fspts on an already trained yolo network.\n\
    test  -> test fspt predictions.\n\
    valid -> validate fspt.\n\
And :\n\
    <datacfg>   -> path to the data configuration file.\n\
    <netcfg>    -> path to the network configuration file.\n\
    [weights]   -> path to a weightfile corresponding to the netcfg file.\n\
                   (optional)\n\
    [inputfile] -> path to a file with list of test images.\n\
                   only for commande test. (optional)\n\
Options are :\n\
    -yolo_thresh -> yolo detection threashold. default 0.5.\n\
    -fspt_thresh -> fspt rejection threshold. default 0.5.\n\
    -hier        -> unused.\n\
    -gpus        -> coma separated list of gpus.\n\
    -out         -> ouput file for test and valid.\n\
    -clear       -> if set, the training number of seen images is reset.\n\
    -refit       -> if set, the fspts are refitted if they already exist.\n\
    -ordered     -> if set, the data are selected sequentialy and not randomly.\n\
    -merge       -> if set, newly extracted data are merged to existing.\n\
    -only_fit    -> if set, don't extract new data. implies -merge.\n\
    -one_thread  -> if set, the fspts are fitted in only one thread instead of\n\
                    one thread per fspt.\n\
    -fullscreen  -> unused.\n\
    -print_stats -> if set, print the statistics of the fspts after training.\n",
                argv[0], argv[1]);
        return;
    }

    float yolo_thresh = find_float_arg(argc, argv, "-yolo_thresh", .5);
    float fspt_thresh = find_float_arg(argc, argv, "-fspt_thresh", .5);
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    int clear = find_arg(argc, argv, "-clear");
    int refit_fspts = find_arg(argc, argv, "-refit");
    int ordered = find_arg(argc, argv, "-ordered");
    int one_thread = find_arg(argc, argv, "-one_thread");
    int only_fit = find_arg(argc, argv, "-only_fit");
    int merge = find_arg(argc, argv, "-merge") || only_fit;
    int print_stats_after_fit = find_arg(argc, argv, "-print_stats");
    int fullscreen = find_arg(argc, argv, "-fullscreen");

    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "test"))
        test_fspt(datacfg, cfg, weights, filename, yolo_thresh, fspt_thresh,
                hier_thresh, outfile, fullscreen);
    else if(0==strcmp(argv[2], "train"))
        train_fspt(datacfg, cfg, weights, gpus, ngpus, clear, refit_fspts,
                ordered, one_thread, merge, only_fit, print_stats_after_fit);
    else if(0==strcmp(argv[2], "valid"))
        validate_fspt(datacfg, cfg, weights, yolo_thresh, fspt_thresh,
                hier_thresh, ngpus, ordered, outfile);
    else if(0==strcmp(argv[2], "recall"))
        validate_fspt_recall(cfg, weights);
    else if (0 == strcmp(argv[2], "stats"))
        print_stats(datacfg, cfg, weights, yolo_thresh, fspt_thresh);
}
