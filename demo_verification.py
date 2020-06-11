from utils import *


if __name__ == '__main__':
    model_pth = 'r2d2_WASF_N16.pt'
    img_pth = 'brooklyn.png'
    net = load_network(model_pth)

    print('Running keypoints detector and descriptor with original R2D2 network ...')
    [keypoints, descriptors, scores, [W,H]] = extract_keypoints(img_pth, net)

    print('Run descriptor of each keypoint with Patch-R2D2 and compare it with original descriptor using normalized cosine similarity score ...')
    for idx in range(len(keypoints)):
        keypoint = keypoints[idx]
        r2d2_desc = descriptors[idx]
        
        [patch, x0, y0] = get_patch(img_pth, keypoint)

        patch_ts = torch.from_numpy(patch).cuda()
        patch_r2d2_desc = patch_R2D2(patch_ts,x0,y0,net)

        cs_score = r2d2_desc.dot(patch_r2d2_desc)
        print('keypoint index: %i\t location: (%i, %i)\t scale: %i\t\t similarity: %.3f'%(idx, keypoint[1], keypoint[0], keypoint[2], cs_score))