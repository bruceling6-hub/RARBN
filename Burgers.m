%% 1. 加载数据（修改为你的mat文件名）
clear; clc; close all;
load('Burgers_Seed97_Data.mat')

%% 2. 固定配置（严格匹配图中顺序和样式）
% 【核心】固定子图顺序（3行3列，和图中完全一致）
fixed_model_order = {
    'Pure HF',
    'MFBNN',
    'MFDNN',
    'MFRNP',
    'NARGP',
    'QRNN',
    'RobustNN',
    'MFPINN',
    'RARBN'
};

% 【核心】模型样式表（和图中颜色、线型、标记1:1匹配）
style_map = containers.Map();
style_map('Pure HF')    = struct('c', '#023E8A', 'ls', '--', 'marker', '*');
style_map('MFBNN')      = struct('c', '#1f77b4', 'ls', '--', 'marker', 'o');
style_map('MFDNN')      = struct('c', '#ff7f0e', 'ls', '--', 'marker', 's');
style_map('MFRNP')      = struct('c', '#2ca02c', 'ls', '-',  'marker', 'd');
style_map('NARGP')      = struct('c', '#17becf', 'ls', '-.', 'marker', '*');
style_map('QRNN')       = struct('c', '#e377c2', 'ls', '-.', 'marker', 'x');
style_map('RobustNN')   = struct('c', '#7f7f7f', 'ls', '--', 'marker', 'd');
style_map('MFPINN')     = struct('c', '#bcbd22', 'ls', ':',  'marker', '*');
style_map('RARBN')      = struct('c', '#d62728', 'ls', '-',  'marker', 'D');

% 自动过滤mat文件中实际存在的模型
vars = who;
models_to_plot = {};
for i = 1:length(fixed_model_order)
    m_name = fixed_model_order{i};
    mu_var = [strrep(m_name, ' ', '_'), '_mu'];
    if ismember(mu_var, vars)
        models_to_plot{end+1} = m_name;
    end
end

if isempty(models_to_plot)
    error('未在mat文件中找到匹配的模型变量，请检查变量名是否为「模型名_mu」格式');
end
fprintf('检测到 %d 个模型，按固定顺序绘图...\n', length(models_to_plot));

%% 3. 基础绘图参数
n_models = length(models_to_plot);
cols = 3;
rows = ceil(n_models / cols);

% 基础数据提取
x_axis = X_plot;
x_min = -1;  % 强制和图中一致：X轴范围-1到1
x_max = 1;

% Marker稀疏度（和图中标记密度一致）
mark_step = max(1, floor(length(x_axis) / 12));
mark_ind = 1:mark_step:length(x_axis);

% 【和图中完全一致】ID/OOD边界：X<0为ID区（白色），X>0为OOD区（浅橙色）
id_start = -1;
id_end = 0;

%% 4. 创建画布与子图
fig = figure('Position', [50, 50, 1200, 1000], 'Color', 'w'); % 画布尺寸匹配图中比例
tlo = tiledlayout(rows, cols, 'Padding', 'compact', 'TileSpacing', 'compact');

% 图例收集变量
legend_handles = [];
legend_labels = {};
last_ax = [];

%% 5. 循环绘制每个子图
for i = 1:n_models
    m_name = models_to_plot{i};
    st = style_map(m_name);
    
    ax = nexttile(tlo);
    last_ax = ax;
    hold on; grid on; box on;
    
    % 获取当前模型数据
    mu_var = [strrep(m_name, ' ', '_'), '_mu'];
    std_var = [strrep(m_name, ' ', '_'), '_std'];
    mu = evalin('base', mu_var);
    std = [];
    if exist(std_var, 'var'); std = evalin('base', std_var); end
    
    % 【强制对齐】X轴范围和刻度
    xlim(ax, [x_min, x_max]);
    xticks(ax, [-1, -0.5, 0, 0.5, 1]);
    
    % --- [修改 1] 智能计算 ID/OOD 边界 ---
    % 逻辑：训练数据覆盖的地方是ID，没覆盖的是OOD
    id_start = x_min;
    id_end = x_max; % 默认全是ID
    if exist('X_train', 'var')
        train_max = max(X_train);
        % 如果训练数据最大值小于0.9（即没覆盖到1），则从那里开始算OOD
        if train_max < 0.9
            id_end = train_max; 
        end
    end
    
    % --- [修改 2] 调整顺序：先画所有曲线，让MATLAB确定Y轴范围 ---
    
    % 1. Ground Truth（黑色实线）
    if exist('y_true', 'var') && ~isempty(y_true)
        plot(x_axis, y_true, 'k-', 'LineWidth', 2, 'HandleVisibility', 'off');
    end
    % 2. LF（灰色虚线）
    if exist('y_lf', 'var') && ~isempty(y_lf)
        plot(x_axis, y_lf, 'k--', 'LineWidth', 1.5, 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
    end
    % 3. 模型预测主线
    plot(x_axis, mu, 'Color', hex2rgb(st.c), ...
         'LineStyle', st.ls, 'LineWidth', 1.5, ...
         'Marker', st.marker, 'MarkerIndices', mark_ind, ...
         'MarkerSize', 6, 'HandleVisibility', 'off');
    % 4. HF训练样本（黑色叉号）
    if exist('X_train', 'var') && exist('y_train_hf', 'var')
        scatter(X_train, y_train_hf, 30, 'k', 'x', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    end
    
    % --- [修改 3] 最后画背景色块：此时Y轴已确定，色块能撑满 ---
    y_lim = get(ax, 'YLim'); % 获取当前最终的Y轴范围
    
    % 绘制 ID 区域 (白色，放在最下面)
    h_id = fill([id_start, id_end, id_end, id_start], ...
                [y_lim(1), y_lim(1), y_lim(2), y_lim(2)], ...
                'w', 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    uistack(h_id, 'bottom'); % 确保在最底层
    
    % 绘制 OOD 区域 (浅橙色，越长越好)
    if id_end < x_max - 1e-9
        h_ood = fill([id_end, x_max, x_max, id_end], ...
                     [y_lim(1), y_lim(1), y_lim(2), y_lim(2)], ...
                     [1, 0.933, 0.894], 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        uistack(h_ood, 'bottom');
    end
    
    % --- C. 子图样式设置 ---
    title(ax, m_name, 'FontWeight', 'bold', 'FontSize', 12, 'Color', 'k');
    ax.FontSize = 10;
    set(ax, 'GridLineStyle', ':', 'GridAlpha', 0.3);
    hold off;
end

%% 6. 绘制统一图例（强制放在正下方-0.2位置）
hold(last_ax, 'on');
vh = [];
vl = {};

% 1. 先添加所有模型（按固定顺序）
for i = 1:length(models_to_plot)
    m_name = models_to_plot{i};
    st = style_map(m_name);
    h = plot(last_ax, NaN, NaN, 'Color', hex2rgb(st.c), ...
             'LineStyle', st.ls, 'LineWidth', 1.5, ...
             'Marker', st.marker, 'MarkerSize', 6);
    vh = [vh, h];
    vl = [vl, m_name];
end

% 2. 再添加固定项
if exist('y_true', 'var') && ~isempty(y_true)
    h = plot(last_ax, NaN, NaN, 'k-', 'LineWidth', 2);
    vh = [vh, h]; vl = [vl, 'Ground truth'];
end
if exist('y_lf', 'var') && ~isempty(y_lf)
    h = plot(last_ax, NaN, NaN, 'k--', 'LineWidth', 1.5, 'Color', [0.5 0.5 0.5]);
    vh = [vh, h]; vl = [vl, 'LF'];
end
if exist('X_train', 'var') && exist('y_train_hf', 'var')
    h = scatter(last_ax, NaN, NaN, 30, 'k', 'x', 'LineWidth', 1.2);
    vh = [vh, h]; vl = [vl, 'HF Samples'];
end
hold(last_ax, 'off');

% --- 【关键修改 1】给图例腾空间：增大底部边距 ---
% 注意：必须在创建图例前设置 tlo 的 Padding
tlo.Padding = 'loose'; % 先设为宽松模式
tlo.TileSpacing = 'compact';

% --- 【关键修改 2】创建图例并强制定位 ---
n_items = length(vh);
ncols_leg = 6; % 保持6列

leg = legend(last_ax, vh, vl, ...
       'NumColumns', ncols_leg, ...
       'FontSize', 10, ...
       'Box', 'off', ...
       'Interpreter', 'none');

% 设置图例位置（归一化单位）
% 格式: [x_left, y_bottom, width, height]
leg.Units = 'normalized';
leg_width = 0.8;   % 图例总宽度 (0.8表示占画布的80%)
leg_height = 0.06; % 图例高度
x_pos = 0.5 - leg_width/2; % 居中计算
y_pos = 0;       % 你要求的 -0.2 位置

set(leg, 'Position', [x_pos, y_pos, leg_width, leg_height]);

% --- 【关键修改 3】调整画布大小，确保下方图例能被看见 ---
% 重新调整 figure 高度，把下方的图例包含进来
fig_pos = get(fig, 'Position');
set(fig, 'Position', [fig_pos(1), fig_pos(2)+100, fig_pos(3), fig_pos(4)+150]); % 增加高度

%% 7. 保存图片
task_name = 'Burgers_1D_Seed97_Reviewer';
filename = sprintf('%s_Fit_Plot_Matched.svg', task_name);
print(fig, filename, '-dsvg', '-r600');
fprintf('绘图完成！已保存至: %s\n', filename);

%% 辅助函数：Hex颜色转RGB（兼容所有MATLAB版本）
function rgb = hex2rgb(hex_str)
    if startsWith(hex_str, '#')
        hex_str = hex_str(2:end);
    end
    r = hex2dec(hex_str(1:2))/255;
    g = hex2dec(hex_str(3:4))/255;
    b = hex2dec(hex_str(5:6))/255;
    rgb = [r, g, b];
end