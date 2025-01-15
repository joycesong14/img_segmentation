def dice_score(pred, origin):
    pred = torch.where(pred > 0.5, 1.0, 0.0)
    origin = torch.where(origin > 0.5, 1.0, 0.0)
    numerator = torch.sum(pred * origin)
    denominator = torch.sum(pred) + torch.sum(origin)  
    dice_score = 2 * numerator/(denominator + 1e-6) # avoid the denominator is 0
    return float(dice_score.detach().to("cpu"))


def train():
    #prepare data
    train_ds, eval_ds = read_data("/home/joyce/dl_lab/2_image_segmentation/montgomery.csv")
    train_data = MontData(train_ds, parent_path="/root/fastestimator_data/Montgomery")
    eval_data = MontData(eval_ds, parent_path="/root/fastestimator_data/Montgomery")
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers= 16)
    eval_loader = DataLoader(eval_data, batch_size=2, shuffle=False, num_workers= 16)

    #create network, loss, optmizer
    model = UNet(input_size=(1, 512, 512), output_channel=2)
    epoches = 20
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCELoss()
    step = 0
    best_dice = float('-inf')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_weights_dir = "/home/joyce/dl_lab/2_image_segmentation/save_dir"

    model.to(device)
    for epoch in range(epoches):
        print("epoch :{}".format(epoch))
        print("start training......")
        model.train()
        for image_train, mask_train in train_loader:
            opt.zero_grad()
            image_train = image_train.to(device)
            mask_train = mask_train.to(device)
            predict_mask = model(image_train)
            loss = loss_fn(predict_mask, mask_train)
            loss.backward()
            opt.step()
            step = step + 1
            if step % 5 == 1:
                print("Epoch: {}, step: {}, loss {}".format(epoch, step, loss))
        print("start evaluation...")

        model.eval()
        dice_scores_left = []
        dice_scores_right = []
        for image_eval, mask_eval in eval_loader:
            image_eval = image_eval.to(device)
            mask_eval = mask_eval.to(device)
            pred_mask = model(image_eval) # B, 2, h, w
            for pred_sample, sample in zip(pred_mask, mask_eval):
                # 2, H, W
                left_pred = pred_sample[0] # H,W
                left_sample = sample[0]
                left_score = dice_score(left_pred, left_sample)

                right_pred = pred_sample[1] # H,W
                right_sample = sample[1]
                right_score = dice_score(right_pred, right_sample)

                dice_scores_left.append(left_score)
                dice_scores_right.append(right_score)

        left_average_score = np.mean(dice_scores_left)
        right_average_score = np.mean(dice_scores_right)  
        final_score =  (left_average_score + right_average_score)/2  
        if final_score > best_dice:
            best_dice = final_score
            model_path = os.path.join(save_weights_dir, "weights.th")   
            torch.save(model.state_dict(), model_path)  
        print("Epoch: {}, left_dice:{}, right_dice:{}, best_dice:{}".format(epoch, left_average_score, right_average_score,best_dice))      
