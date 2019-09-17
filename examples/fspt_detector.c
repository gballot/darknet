#include "darknet.h"

void test_fspt_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
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
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        //printf("%d\n", nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        //TODO GAB
        /*
        printf("TODO GAB : GET THE INPUT FOR FSPT\n");
        layer small_obj_layer = net->layers[81];
        layer medium_obj_layer = net->layers[93];
        layer big_obj_layer = net->layers[105];
        layer small_yolo = net->layers[82];
        layer medium_yolo = net->layers[94];
        layer big_yolo = net->layers[106];
        int i,j;                                                                    

        for(i = 0; i < nboxes; ++i){
          char labelstr[4096] = {0};
          int class = -1;
          for(j = 0; j < medium_yolo.classes; ++j){
            if (dets[i].prob[j] > thresh){
              if (class < 0) {
                strcat(labelstr, names[j]);
                class = j;
              } else {
                strcat(labelstr, ", ");
                strcat(labelstr, names[j]);
              }
              printf("%s (class %d): %.0f%% - box(x,y,w,h) : %f,%f,%f,%f\n", names[j], j, dets[i].prob[j]*100, dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.h, dets[i].bbox.h);          
            }
          }
        }
        //get_yolo_detection(big_yolo, 0, 0, net.w, net.h, thresh, 
        */
        // END TODO
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
