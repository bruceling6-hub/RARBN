clear; clc; close all;
load('Burgers_Seed97_Data.mat')
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
vars = who;
models_to_plot = {};
for i = 1:length(fixed_model_order)
    m_name = fixed_model_order{i};
    mu_var = [strrep(m_name, ' ', '_'), '_mu'];
    if ismember(mu_var, vars)
        models_to_plot{end+1} = m_name;
    end
end
n_models = length(models_to_plot);
cols = 3;
rows = ceil(n_models / cols);
x_axis = X_plot;
x_min = -1;  
x_max = 1;
mark_step = max(1, floor(length(x_axis) / 12));
mark_ind = 1:mark_step:length(x_axis);
id_start = -1;
id_end = 0;
fig = figure('Position', [50, 50, 1200, 1000], 'Color', 'w'); 
tlo = tiledlayout(rows, cols, 'Padding', 'compact', 'TileSpacing', 'compact');
legend_handles = [];
legend_labels = {};
last_ax = [];
for i = 1:n_models
    m_name = models_to_plot{i};
    st = style_map(m_name);
    ax = nexttile(tlo);
    last_ax = ax;
    hold on; grid on; box on;
    mu_var = [strrep(m_name, ' ', '_'), '_mu'];
    std_var = [strrep(m_name, ' ', '_'), '_std'];
    mu = evalin('base', mu_var);
    std = [];
    if exist(std_var, 'var'); std = evalin('base', std_var); end
    xlim(ax, [x_min, x_max]);
    xticks(ax, [-1, -0.5, 0, 0.5, 1]);
    id_start = x_min;
    id_end = x_max; 
    if exist('X_train', 'var')
        train_max = max(X_train);
        if train_max < 0.9
            id_end = train_max; 
        end
    end
    if exist('y_true', 'var') && ~isempty(y_true)
        plot(x_axis, y_true, 'k-', 'LineWidth', 2, 'HandleVisibility', 'off');
    end
    if exist('y_lf', 'var') && ~isempty(y_lf)
        plot(x_axis, y_lf, 'k--', 'LineWidth', 1.5, 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
    end
    plot(x_axis, mu, 'Color', hex2rgb(st.c), ...
         'LineStyle', st.ls, 'LineWidth', 1.5, ...
         'Marker', st.marker, 'MarkerIndices', mark_ind, ...
         'MarkerSize', 6, 'HandleVisibility', 'off');
    if exist('X_train', 'var') && exist('y_train_hf', 'var')
        scatter(X_train, y_train_hf, 30, 'k', 'x', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    end
    y_lim = get(ax, 'YLim'); 
    h_id = fill([id_start, id_end, id_end, id_start], ...
                [y_lim(1), y_lim(1), y_lim(2), y_lim(2)], ...
                'w', 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    uistack(h_id, 'bottom');
    if id_end < x_max - 1e-9
        h_ood = fill([id_end, x_max, x_max, id_end], ...
                     [y_lim(1), y_lim(1), y_lim(2), y_lim(2)], ...
                     [1, 0.933, 0.894], 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        uistack(h_ood, 'bottom');
    end
    title(ax, m_name, 'FontWeight', 'bold', 'FontSize', 12, 'Color', 'k');
    ax.FontSize = 10;
    set(ax, 'GridLineStyle', ':', 'GridAlpha', 0.3);
    hold off;
end
hold(last_ax, 'on');
vh = [];
vl = {};
for i = 1:length(models_to_plot)
    m_name = models_to_plot{i};
    st = style_map(m_name);
    h = plot(last_ax, NaN, NaN, 'Color', hex2rgb(st.c), ...
             'LineStyle', st.ls, 'LineWidth', 1.5, ...
             'Marker', st.marker, 'MarkerSize', 6);
    vh = [vh, h];
    vl = [vl, m_name];
end
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
tlo.Padding = 'loose'; 
tlo.TileSpacing = 'compact';
n_items = length(vh);
ncols_leg = 6;
leg = legend(last_ax, vh, vl, ...
       'NumColumns', ncols_leg, ...
       'FontSize', 10, ...
       'Box', 'off', ...
       'Interpreter', 'none');
leg.Units = 'normalized';
leg_width = 0.8;  
leg_height = 0.06; 
x_pos = 0.5 - leg_width/2; 
y_pos = 0; 
set(leg, 'Position', [x_pos, y_pos, leg_width, leg_height]);
fig_pos = get(fig, 'Position');
set(fig, 'Position', [fig_pos(1), fig_pos(2)+100, fig_pos(3), fig_pos(4)+150]);
task_name = 'Burgers_1D_Seed97_Reviewer';
filename = sprintf('%s_Fit_Plot_Matched.svg', task_name);
print(fig, filename, '-dsvg', '-r600');
function rgb = hex2rgb(hex_str)
    if startsWith(hex_str, '#')
        hex_str = hex_str(2:end);
    end
    r = hex2dec(hex_str(1:2))/255;
    g = hex2dec(hex_str(3:4))/255;
    b = hex2dec(hex_str(5:6))/255;
    rgb = [r, g, b];
end
