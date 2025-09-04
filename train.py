#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.visualization import RealTimeHistogramVisualizer, BarHistogramVisualizer, Simple2DHistogramVisualizer
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, enable_visualization=True):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 실시간 히스토그램 시각화 초기화
    histogram_viz = None
    bar_viz = None
    simple_viz = None
    if enable_visualization:
        try:
            # 간단한 2D 버전 (권장) - 초기 p_max를 5로 설정
            simple_viz = Simple2DHistogramVisualizer(max_history=200, update_interval=25, p_min=1.0, p_max=5.0, num_bins=30)
            
            # Surface plot 스타일 (선택적) - 전체 iteration 데이터 저장
            histogram_viz = RealTimeHistogramVisualizer(max_history=100, update_interval=100, p_min=1.0, p_max=5.0, num_bins=32, store_all_data=True)
            
            # Bar plot 스타일 (선택적, 리소스 집약적)
            # bar_viz = BarHistogramVisualizer(max_history=50, update_interval=300, p_min=1.0, p_max=5.0, num_bins=25)
            
            print("Real-time histogram visualization enabled (2D + 3D with full iteration history)")
        except Exception as e:
            print(f"Failed to initialize histogram visualization: {e}")
            histogram_viz = None
            bar_viz = None
            simple_viz = None
    
    # 초기 Shape 파라미터 정보 출력 (alpha: p-norm parameter)
    print("\n=== Initial Shape Parameters (p-norm alpha) ===")
    initial_shapes = gaussians.get_shape
    initial_alpha = initial_shapes[:, 0]
    # alpha를 p값으로 변환: p = 1 + (p_max - 1) * sigmoid(alpha)
    p_max = 64.0
    sigmoid_alpha = torch.sigmoid(initial_alpha)
    p_values = 1.0 + (p_max - 1.0) * sigmoid_alpha
    print(f"Alpha - Mean: {initial_alpha.mean():.4f}, Std: {initial_alpha.std():.4f}, Min: {initial_alpha.min():.4f}, Max: {initial_alpha.max():.4f}")
    print(f"P-norm (computed) - Mean: {p_values.mean():.4f}, Std: {p_values.std():.4f}, Min: {p_values.min():.4f}, Max: {p_values.max():.4f}")
    print(f"Total Gaussians: {len(initial_shapes)}")
    print("=================================\n")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                # Shape 파라미터 통계 계산 (alpha와 p-norm 중심)
                shapes = gaussians.get_shape
                alpha = shapes[:, 0]
                # alpha를 p값으로 변환
                p_max = 64.0
                sigmoid_alpha = torch.sigmoid(alpha)
                p_values = 1.0 + (p_max - 1.0) * sigmoid_alpha
                alpha_mean = alpha.mean()
                alpha_std = alpha.std()
                p_mean = p_values.mean()

                # 실시간 히스토그램 시각화 데이터 추가 및 업데이트
                # 2D 간단한 시각화 (우선순위)
                if simple_viz is not None:
                    simple_viz.add_data(iteration, p_values)
                    if simple_viz.should_update(iteration):
                        try:
                            simple_viz.update_plot()
                        except Exception as e:
                            print(f"Simple visualization error: {e}")
                
                # 3D Surface 시각화
                if histogram_viz is not None:
                    histogram_viz.add_data(iteration, p_values)
                    if histogram_viz.should_update(iteration):
                        try:
                            histogram_viz.update_plot()
                        except Exception as e:
                            print(f"3D histogram visualization error: {e}")
                
                # 3D Bar 시각화 (선택적)
                if bar_viz is not None:
                    bar_viz.add_data(iteration, p_values)
                    if bar_viz.should_update(iteration):
                        try:
                            bar_viz.update_plot()
                        except Exception as e:
                            print(f"Bar histogram visualization error: {e}")

                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                    "α": f"{alpha_mean:.3f}±{alpha_std:.3f}",
                    "p": f"{p_mean:.2f}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Shape 파라미터 상세 출력 (매 100 iteration마다)
            if iteration % 100 == 0:
                shapes = gaussians.get_shape
                alpha = shapes[:, 0]
                # alpha를 p값으로 변환
                p_max = 64.0
                sigmoid_alpha = torch.sigmoid(alpha)
                p_values = 1.0 + (p_max - 1.0) * sigmoid_alpha
                
                alpha_mean_ = alpha.mean().item()
                alpha_std_ = alpha.std().item()
                alpha_min_ = alpha.min().item()
                alpha_max_ = alpha.max().item()
                
                p_mean_ = p_values.mean().item()
                p_std_ = p_values.std().item()
                p_min_ = p_values.min().item()
                p_max_actual = p_values.max().item()

                # p값 분포: p=1(L1), p=2(L2), p>>1(L∞)
                l1_like = torch.sum(p_values < 1.5).item()  # p < 1.5: L1에 가까움
                l2_like = torch.sum((p_values >= 1.5) & (p_values < 3.0)).item()  # 1.5 ≤ p < 3.0: L2에 가까움
                linf_like = torch.sum(p_values >= 3.0).item()  # p ≥ 3.0: L∞에 가까움
                n = len(p_values)

                print(f"\n[ITER {iteration}] Shape P-norm Statistics:")
                print(f"  Alpha - Mean: {alpha_mean_:.4f}, Std: {alpha_std_:.4f}, Min: {alpha_min_:.4f}, Max: {alpha_max_:.4f}")
                print(f"  P-norm - Mean: {p_mean_:.4f}, Std: {p_std_:.4f}, Min: {p_min_:.4f}, Max: {p_max_actual:.4f}")
                print(f"  P-norm Distribution:")
                print(f"    L1-like (p < 1.5): {l1_like} ({100*l1_like/n:.1f}%)")
                print(f"    L2-like (1.5 ≤ p < 3.0): {l2_like} ({100*l2_like/n:.1f}%)")
                print(f"    L∞-like (p ≥ 3.0): {linf_like} ({100*linf_like/n:.1f}%)")

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                
                # Shape 파라미터 텐서보드 로깅 (alpha와 p-norm 중심)
                if iteration % 10 == 0:
                    tb_writer.add_scalar('shape_stats/alpha_mean', alpha_mean, iteration)
                    tb_writer.add_scalar('shape_stats/alpha_std', alpha_std, iteration)
                    tb_writer.add_scalar('shape_stats/p_mean', p_mean, iteration)
                    tb_writer.add_scalar('shape_stats/alpha_min', alpha.min(), iteration)
                    tb_writer.add_scalar('shape_stats/alpha_max', alpha.max(), iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None
    
    # 최종 Shape 파라미터 정보 출력 (alpha와 p-norm 중심)
    print("\n=== Final Shape Parameters (p-norm alpha) ===")
    final_shapes = gaussians.get_shape
    final_alpha = final_shapes[:, 0]
    # alpha를 p값으로 변환
    p_max = 64.0
    sigmoid_final_alpha = torch.sigmoid(final_alpha)
    final_p_values = 1.0 + (p_max - 1.0) * sigmoid_final_alpha
    
    final_p_max_actual = final_p_values.max().item()
    
    print(f"Final Alpha - Mean: {final_alpha.mean():.4f}, Std: {final_alpha.std():.4f}, Min: {final_alpha.min():.4f}, Max: {final_alpha.max():.4f}")
    print(f"Final P-norm - Mean: {final_p_values.mean():.4f}, Std: {final_p_values.std():.4f}, Min: {final_p_values.min():.4f}, Max: {final_p_max_actual:.4f}")

    # 최종 분포 요약
    l1_like = torch.sum(final_p_values < 1.5).item()
    l2_like = torch.sum((final_p_values >= 1.5) & (final_p_values < 3.0)).item()
    linf_like = torch.sum(final_p_values >= 3.0).item()
    n_final = len(final_p_values)
    print(f"Final P-norm Distribution:")
    print(f"  L1-like (p < 1.5): {l1_like} ({100*l1_like/n_final:.1f}%)")
    print(f"  L2-like (1.5 ≤ p < 3.0): {l2_like} ({100*l2_like/n_final:.1f}%)")
    print(f"  L∞-like (p ≥ 3.0): {linf_like} ({100*linf_like/n_final:.1f}%)")
    print(f"Total Final Gaussians: {len(final_shapes)}")
    print(f"Actual P-norm range: 1.0 - {final_p_max_actual:.4f}")
    print("==============================\n")
    
    # 최종 히스토그램 저장
    if simple_viz is not None:
        try:
            final_save_path = os.path.join(dataset.model_path, "final_p_histogram_2d.png")
            simple_viz.save_final_plot(final_save_path)
            simple_viz.close()
        except Exception as e:
            print(f"Failed to save final 2D histogram: {e}")
    
    if histogram_viz is not None:
        try:
            final_save_path = os.path.join(dataset.model_path, "final_p_histogram_3d.png")
            histogram_viz.save_final_plot(final_save_path)
            histogram_viz.close()
        except Exception as e:
            print(f"Failed to save final 3D histogram: {e}")
    
    if bar_viz is not None:
        try:
            bar_viz.close()
        except Exception as e:
            print(f"Failed to close bar visualization: {e}")
    
    return gaussians

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--disable_visualization", action="store_true", help="Disable real-time histogram visualization")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    gaussians = training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, enable_visualization=not args.disable_visualization)

    # All done
    print("\nTraining complete.")
