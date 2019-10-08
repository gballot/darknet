#include "darknet.h"

#include <stdlib.h>

#include "utils.h"

void test_fspt(char *datacfg, char *cfgfile, char *weightfile, char *filename, float yolo_thresh, float fspt_thresh, float hier_thresh, char *outfile, int fullscreen)
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
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n-1];


        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, yolo_thresh, hier_thresh, 0, 1, &nboxes);
        //printf("%d\n", nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        draw_detections(im, dets, nboxes, yolo_thresh, names, alphabet, l.classes);
        dets = get_network_fspt_boxes(net, im.w, im.h, yolo_thresh, fspt_thresh, hier_thresh, 0, 1, &nboxes);
        free_detections(dets, nboxes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            make_window("predictions", 512, 512, 0);
            show_image(im, "predictions", 0);
#endif
        }

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}

void train_fspt(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear) {
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
    for(i = 0; i < ngpus; ++i){
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
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    data train, buffer;

    //TODO: take a fspt layer instead of the yolo one.
    layer l = net->layers[net->n - 2];

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = get_base_args(net);
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    //args.type = INSTANCE_DATA;
    args.threads = 64;

    pthread_t load_thread = load_data(args);
    double time;
    while(get_current_batch(net) < net->max_batches){
        time=what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        /*
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           if(!b.x) break;
           printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
           }
         */
        /*
           int zz;
           for(zz = 0; zz < train.X.cols; ++zz){
           image im = float_to_image(net->w, net->h, 3, train.X.vals[zz]);
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[zz] + k*5, 1);
           printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
           draw_bbox(im, b, 1, 1,0,0);
           }
           show_image(im, "truth11");
           cvWaitKey(0);
           save_image(im, "truth11");
           }
         */

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
        printf("%ld: %f rate, %lf seconds, %d images\n", get_current_batch(net), get_current_rate(net), what_time_is_it_now()-time, i*imgs);
        if(i%100==0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
        if(i%10000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    fprintf(stderr, "Data extraction done... Fitting FSPTs\n");
    for (int i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == FSPT) {
            for (int class = 0; class < l.classes; ++class) {
                fspt_t *fspt = l.fspts[class];
                int n = l.fspt_n_training_data[class];
                float *X = l.fspt_training_data[class];
                criterion_args *args = calloc(1, sizeof(criterion_args)); 
                /*TODO: Theses values can be passed from up frame */
                args->max_try_p = 1.;
                args->max_feature_p = 1.;
                args->thresh = 0.01;
                fspt_fit(n, X, args, fspt);
                char buff[256];
                sprintf(buff, "%s/%s_fspt_class_%d.dat",
                        backup_directory, base, class);
                int succ = 1;
                fspt_save(buff, *fspt, &succ);
                if (!succ) fprintf(stderr, "Save fspt faild...");
                print_fspt(fspt);
            }
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

void validate_fspt(char *datacfg, char *cfgfile, char *weightfile, char *outfile) {
    //TODO
    error("TODO");
}

void validate_fspt_recall(char *cfgfile, char *weightfile) {
    //TODO
    error("TODO");
}

void run_fspt(int argc, char **argv)
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float yolo_thresh = find_float_arg(argc, argv, "-yolo_thresh", .5);
    float fspt_thresh = find_float_arg(argc, argv, "-fspt_thresh", .5);
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int avg = find_int_arg(argc, argv, "-avg", 3);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
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

    int clear = find_int_arg(argc, argv, "-clear", 1);
    int fullscreen = find_arg(argc, argv, "-fullscreen");
    int width = find_int_arg(argc, argv, "-w", 0);
    int height = find_int_arg(argc, argv, "-h", 0);
    int fps = find_int_arg(argc, argv, "-fps", 0);
    //int class = find_int_arg(argc, argv, "-class", 0);

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "test")) test_fspt(datacfg, cfg, weights, filename, yolo_thresh, fspt_thresh, hier_thresh, outfile, fullscreen);
    else if(0==strcmp(argv[2], "train")) train_fspt(datacfg, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "valid")) validate_fspt(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "recall")) validate_fspt_recall(cfg, weights);
}
