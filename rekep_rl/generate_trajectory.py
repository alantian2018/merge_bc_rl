import argparse
import os
from util_rlkit.playback_dataset_custom import playback_dataset
from datetime import datetime

## generates robomimic playback trajectory

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
   # dt = datetime.now().strftime('%Y-%m-%d-%H:%M')
    parser.add_argument('--name', type=str, required=True, help='name of task')
    parser.add_argument('--dataset', type=str, required=True, help='path to hdf5 dataset file')
    parser.add_argument(
        '--camera_width',
        default=1024
    )
    parser.add_argument(
        '--camera_height',
        default=1024
    ) 
    args = parser.parse_args()
   
    file_dir  = f'./data/{args.name}/' 
    os.makedirs(file_dir,exist_ok=True)

    playback_arg_list = [
    '--dataset', args.dataset,
    '--render_image_names', 'agentview',
    "--video_path", os.path.join(file_dir,'actions.mp4'),
    '--video_skip', '1',
    '--use-actions',
    '--n', '1',
    ]

    playback_args = argparse.ArgumentParser()
 
    playback_args.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    playback_args.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    # number of trajectories to playback. If omitted, playback all of them.
    playback_args.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )

    # Use image observations instead of doing playback using the simulator env.
    playback_args.add_argument(
        "--use-obs",
        action='store_true',
        help="visualize trajectories with dataset image observations instead of simulator",
    )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    playback_args.add_argument(
        "--use-actions",
        action='store_true',
        help="use open-loop action playback instead of loading sim states",
    )

    # Whether to render playback to screen
    playback_args.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the dataset playback to the specified path
    playback_args.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )

    # How often to write video frames during the playback
    playback_args.add_argument(
        "--video_skip",
        type=int,
        default=1,
        help="render frames to video every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    playback_args.add_argument(
        "--render_image_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )

    # depth observations to use for writing to video
    playback_args.add_argument(
        "--render_depth_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) depth observation(s) to use for rendering to video"
    )

    # Only use the first frame of each episode
    playback_args.add_argument(
        "--first",
        action='store_true',
        help="use first frame of each episode",
    )
    
    parsed_playback_args = playback_args.parse_args(playback_arg_list)

    playback_dataset(parsed_playback_args, args.camera_height,args.camera_width)