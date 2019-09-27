#include "darknet.h"

void test_fspt_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float yolo_thresh, float fspt_thresh, float hier_thresh, char *outfile, int fullscreen)
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

void run_fspt(int argc, char **argv)
{
    float yolo_thresh = find_float_arg(argc, argv, "-yolo_thresh", .5);
    float fspt_thresh = find_float_arg(argc, argv, "-fspt_thresh", .5);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train_yolo/train_fspt/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    int avg = find_int_arg(argc, argv, "-avg", 1);
    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "test")) test_fspt(cfg, weights, filename, thresh);
    else if(0==strcmp(argv[2], "train_yolo")) train_yolo(cfg, weights);
    else if(0==strcmp(argv[2], "train_fspt")) train_fspt(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_fspt(cfg, weights);
}
