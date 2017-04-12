img_file = '/media/anu-user1/2TB/xin/torch-gan/datasets/img_align_celeba/';
save_folder_HR = '/media/anu-user1/2TB/xin/Data/face_HR_frontal/';
save_folder_LR = '/media/anu-user1/2TB/xin/Data/face_LR_rotate/';

org_start_x = 35;
org_start_y = 55;
% org_H = 131; H_half = 65;
% org_W = 101; W_half = 50;
% border = 27;
org_H = 133; H_half = 66;
org_W = 107; W_half = 53;
border = 30;

arr_file = dir([img_file,'*.jpg']);
num = size(arr_file,1);
num_data = 35000;
data_HR  = zeros(128,128,3,num_data,'single');
data_LR  = zeros(16,16,3,num_data,'single');

attr_mat = load('list_attr_celeba_clean.mat');
attr = attr_mat.attr;

%% HR
% i = 1;
% j = 1;
% % for i = 1:num_data
% while j<num_data
%     filename = arr_file(i).name;
%     if attr(i,11) == 1 | attr(i,16) == 1 | attr(i,36) == 1
%         i = i+1;
%         continue;
%     end
%     
%     img = im2double(imread([img_file filename]));
%     [h,w,d] = size(img);
%     crop = img(org_start_y:org_start_y+org_H-1, org_start_x:org_start_x+org_W-1,:);
%     img_hr = imresize(crop,[128,128],'bicubic');
%     img_hr = single(img_hr(:,:,[3,2,1]));
%     data_HR(:,:,:,j) = img_hr;
%     i = i+1;
%     j = j+1;
%     %imwrite(img_hr,[save_folder_HR filename(1:end-4) '.png'],'png');
% end
% 
% data_HR = permute(data_HR,[2,1,3,4]);
% h5create('YTC_HR_rotate_v6.hdf5','/YTC',[128 128 3 num_data]);
% h5disp('YTC_HR_rotate_v6.hdf5');
% h5write('YTC_HR_rotate_v6.hdf5','/YTC',data_HR);

%% LR
rng(1)
i = 1;
j = 1;
while j < num_data
%for i = 1:num_data
    filename = arr_file(i).name;
    if attr(i,11) == 1 | attr(i,16) == 1 | attr(i,36) == 1
        i = i+1;
        continue;
    end
    
    img = im2double(imread([img_file filename]));
    scale = rand(1)*0.2+0.8; % org: 0.85
    [m,n,d] = size(img);
    c_m = floor(m/2);
    c_n = floor(n/2);
    
    img_s = imresize(img,scale,'bicubic');
    [m_s,n_s,d_s] = size(img_s);
    c_m_s = floor(m_s/2);
    c_n_s = floor(n_s/2);
    
    img = zeros(size(img));
    st_y = c_m - c_m_s +1;
    st_x = c_n - c_n_s +1;
    
%     img(st_y:st_y+m_s-1, st_x:st_x+n_s-1,:) = img_s;
%     crop = img(org_start_y-border:org_start_y+org_H-1+border, org_start_x-border:org_start_x+org_W-1+border,:);
    theta = (rand(1)-0.5)*90;
    crop = imrotate(img_s,theta,'bicubic','crop');
    [h,w,d] = size(crop);
    ry = fix((rand(1)*h-h/2)*0.1);
    rx = fix((rand(1)*w-w/2)*0.1);
    cy = floor(h/2)+1+ry;
    cx = floor(w/2)+1+rx;
%   H_half = 64;
% 	W_half = 51;
    img_crop = crop(cy-H_half:cy+H_half, cx-W_half: cx+W_half,:);
    img_lr = imresize(img_crop,[16,16],'bicubic');
    img_lr = imnoise(img_lr,'gaussian',0,0.1^2);
%     imshow(img_crop,[])
%     drawnow
%     pause(0.2)
    data_LR(:,:,:,j) = single(img_lr(:,:,[3,2,1]));
    i = i+1;
    j = j+1;
    %imwrite(img_lr,[save_folder_LR filename(1:end-4) '.png'],'png');
end

data_LR = permute(data_LR,[2,1,3,4]);
h5create('YTC_LR_rotate_v6_01.hdf5','/YTC',[16 16 3 num_data]);
h5disp('YTC_LR_rotate_v6_01.hdf5');
h5write('YTC_LR_rotate_v6_01.hdf5','/YTC',data_LR);


%% read hdf5
Data = h5read('YTC_LR_rotate_v6_01.hdf5','/YTC');
Data = permute(Data,[2,1,3,4]);
Data = Data(:,:,[3,2,1],1:100);

for i = 1:100
    img = Data(:,:,:,i);
    img = imresize(img,[128,128]);
    imshow(img,[])
    drawnow
    pause(0.2)
end