import time
import dearpygui.dearpygui as dpg
import tkinter as tk
from numba import njit , prange
import numpy as np
import cv2
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()
dpg.create_context()
viscosity = 1.5e-4
nx , ny = 250 , 250
v_mesh = 1
u_mesh = 1
u = np.zeros((ny,nx))
v = np.zeros((ny,nx))
p = np.zeros((ny,nx))
mask = np.zeros((ny,nx), dtype=bool)
texture_data = np.zeros(ny * nx * 4 , dtype = np.float32)
is_running = False
u_uniform = 50
start_time = time.perf_counter()
dt = 0.01
dx = 0.1
dy = 0.1
emitted_function = False
is_paused = False
image = None
clicker = 0

def image_callback_data(_,file_data):
    global mask , u , v , p , is_running
    global image
    image_path = file_data['file_path_name']
    raw_image = cv2.imread(image_path,0)
    image = cv2.resize(raw_image, (nx, ny - 20))
    _, binary_value = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(binary_value) > (nx * (ny - 20)) / 2:
        binary_value = 1 - binary_value
    full_mask = np.zeros((ny,nx),dtype=bool)
    full_mask[10:ny-10,:] = binary_value.astype(np.bool)
    mask = full_mask
    u = np.zeros((ny,nx))
    v = np.zeros((ny,nx))
    p = np.zeros((ny,nx))
    is_running = True

def update_configurations():
    global emitted_function , u_uniform , viscosity , u_mesh , v_mesh , nx , ny , u , v , p , texture_data ,mask , image
    u_uniform = float(dpg.get_value('velocity_configuration'))
    viscosity = float(dpg.get_value('viscosity_configuration'))
    u_mesh = float(dpg.get_value('speed_mesh_X_configuration'))
    v_mesh = float(dpg.get_value('speed_mesh_Y_configuration'))
    angle = float(dpg.get_value('angle_of_attack_configuration'))
    u = np.zeros((ny, nx), dtype=np.float32)
    v = np.zeros((ny, nx), dtype=np.float32)
    p = np.zeros((ny, nx), dtype=np.float32)
    texture_data = np.zeros(ny * nx * 4, dtype=np.float32)

    if image is not None:
        img_h, img_w = image.shape
        center = (nx // 2 , ny // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center,-angle,1.0)
        rotated_image = cv2.warpAffine(image,rotation_matrix,(img_w,img_h),flags = cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,borderValue=255)
        _ , binary_value = cv2.threshold(rotated_image,0,1,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.sum(binary_value) >= (nx * ny) // 2:
            binary_value = 1 - binary_value
        full_mask = np.zeros((ny,nx),dtype=bool)
        full_mask[10:ny - 10,:] = binary_value.astype(np.bool)
        mask = full_mask
    if dpg.does_item_exist('fluid_texture'):
        dpg.set_value('fluid_texture',list(texture_data))
    dpg.set_value('series_cp_upper',[[],[]])
    dpg.set_value('series_cp_lower',[[],[]])
    dpg.set_value('series_cl_lift',[[],[]])
    dpg.set_value('series_cd_drag',[[],[]])
    dpg.fit_axis_data("X_axis_pressure")
    dpg.fit_axis_data("Y_axis_pressure")
    dpg.fit_axis_data("X_axis_lift")
    dpg.fit_axis_data("Y_axis_lift")
    dpg.fit_axis_data("X_axis_drag")
    dpg.fit_axis_data("Y_axis_drag")
    emitted_function = True
    is_paused = False
    return u_uniform , viscosity , u_mesh , v_mesh

@njit(parallel=True,fastmath=True)
def computational_system(mask,u,v,p,texture_data,u_uniform,viscosity,u_mesh,v_mesh,nx,ny):
    rows , cols = mask.shape
    ny , nx = mask.shape
    div = np.zeros((ny,nx),dtype=np.float32)
    u_new = u.copy()
    v_new = v.copy()
    for i in prange(1,rows-1):
        for j in range(1,cols-1):
            if not mask[i,j]:
                u_new[i, j] = u[i, j] - u_mesh * (dt / dx) * (u[i, j] - u[i, j - 1]) - \
                              v_mesh * (dt / dy) * (u[i, j] - u[i - 1, j]) + \
                              viscosity * (dt / dx ** 2) * (u[i, j - 1] - 2 * u[i, j] + u[i, j + 1]) + \
                              viscosity * (dt / dy ** 2) * (u[i - 1, j] - 2 * u[i, j] + u[i + 1, j])
                v_new[i, j] = v[i, j] - u_mesh * (dt / dx) * (v[i, j] - v[i, j - 1]) - \
                              v_mesh * (dt / dy) * (v[i, j] - v[i - 1, j]) + \
                              viscosity * (dt / dx ** 2) * (v[i, j - 1] - 2 * v[i, j] + v[i, j + 1]) + \
                              viscosity * (dt / dy ** 2) * (v[i - 1, j] - 2 * v[i, j] + v[i + 1, j])
    for i in prange(1, rows - 1):
        for j in range(1, cols - 1):
            div[i, j] = ((u_new[i, j + 1] - u_new[i, j - 1]) / (2 * dx) + (u_new[i + 1, j] - u_new[i - 1, j]) / (
                        2 * dy))
    for _ in range(100):
        p_old = p.copy()
        for i in prange(1,rows - 1):
            for j in range(1,cols - 1):
                p[i, j] = ((p_old[i, j + 1] + p_old[i, j - 1]) * dy ** 2 +
                           (p_old[i + 1, j] + p_old[i - 1, j]) * dx ** 2 -
                           (div[i, j] * dx ** 2 * dy ** 2)) / (2 * (dx ** 2 + dy ** 2))
        p[:,-1] = 0.0
        p[:,0] = p[:,1]
        p[0,:] = p[1,:]
        p[-1,:] = p[-2,:]
    for i in prange(rows):
        for j in range(cols):
            if mask[i, j]:
                p[i, j] = 0.0
    for i in prange(rows - 1):
        for j in range(cols - 1):
            if not mask[i, j]:
                u_new[i, j] -= (dt / (2 * dx)) * (p[i, j + 1] - p[i, j - 1])
                v_new[i, j] -= (dt / (2 * dy)) * (p[i + 1, j] - p[i - 1, j])
            elif mask[i + 1, j] or mask[i - 1, j] or mask[i, j + 1] or mask[i, j - 1]:
                compound_u_y = 0.0
                if mask[i + 1, j]: compound_u_y += -u[i, j] / dy
                if mask[i - 1, j]: compound_u_y += u[i, j] / dy
                compound_u_x = 0.0
                if mask[i, j + 1]: compound_u_x += -u[i, j] / dx
                if mask[i, j - 1]: compound_u_x += u[i, j] / dx
                u_new[i, j] = u[i, j] + viscosity * dt * (compound_u_x / dx + compound_u_y / dy)
                compound_v_y = 0.0
                if mask[i + 1, j]: compound_v_y += -v[i, j] / dy
                if mask[i - 1, j]: compound_v_y += v[i, j] / dy
                compound_v_x = 0.0
                if mask[i, j + 1]: compound_v_x += -v[i, j] / dx
                if mask[i, j - 1]: compound_v_x += v[i, j] / dx
                v_new[i, j] = v[i, j] + viscosity * dt * (compound_v_x / dx + compound_v_y / dy)
            else:
                u_new[i, j] = 0.0
                v_new[i, j] = 0.0
    for i in range(cols):
        u_new[i,0] = u_uniform
        v_new[i,0] = 0.0
        u_new[i,-1] = u_new[i,-2]
        v_new[i,-1] = v_new[i,-2]

    rgba = np.zeros(ny * nx * 4 , dtype=np.float32)
    for i in prange(rows):
        for j in range(cols):
            idx = (i * nx + j) * 4
            if mask[i, j]:
                rgba[idx] = 0.9
                rgba[idx+1] = 0.8
                rgba[idx+2] = 0.7
                rgba[idx+3] = 1.0
            else:
                result_velocity = np.sqrt(u_new[i,j] ** 2 + v_new[i,j] ** 2)
                v_normal = result_velocity / 50.0
                if v_normal > 1.0:
                    v_normal = 1.0
                rgba[idx] = 0.1 * (1.0 - v_normal)
                rgba[idx + 1] = 0.4 * v_normal
                rgba[idx + 2] = 0.6 * v_normal + 0.2
                rgba[idx + 3] = 1.0

    return u_new , v_new , p , rgba

def open_curtains():
    global clicker
    if clicker % 2 == 0:
        dpg.show_item('dialog_window')
        clicker += 1
        print('!')
    elif clicker % 2 == 1:
        dpg.hide_item('dialog_window')
        clicker = 0
        print('?')

def stop_animation(sender,app_data):
    global is_paused
    if not is_paused: is_paused = True
    elif is_paused: is_paused = False
    if is_paused:
        dpg.set_item_label('animation_view',label='Resume simulation')
    else:
        dpg.set_item_label('animation_view',label='Pause simulation')

def save_as_csv():
    import psutil
    import pandas as pd
    from faker import Faker
    fake = Faker('en_US')
    global history_cl , history_cd , history_fps , time_history , quality_history
    disk = psutil.disk_usage('/')
    free_gb = disk.free / (1024 ** 2)
    if is_paused:
        dpg.set_value('warning', value='')
        if free_gb < 1:
            dpg.set_value('warning',value='There is no free memory on your disk')
        else:
            print(len(history_cl))
            print(len(history_cd))
            print(len(history_fps))
            print(len(time_history)) #626
            print(len(quality_history)) #626
            dpg.set_value('warning', value='')
            df = pd.DataFrame({
                'Lift coefficient' : history_cl,
                'Drag coefficient' : history_cd,
                'Frames' : history_fps,
                'Time' : time_history,
                'Aerodynamic Quality' : quality_history
            })
            df.to_csv(f"configuration{fake.bothify('############')}.csv",index=False)
    else:
        dpg.set_value('warning',value=' [You cannot save configurations] ')

def save_as_txt():
    global history_cl , history_cd , history_fps , time_history , quality_history
    from faker import Faker
    fake = Faker('en_US')
    if is_paused:
        dpg.set_value('warning', value='')
        string = f'Lift coefficient : {history_cl[-1]} , Drag coefficient : {history_cd[-1]} , Aerodynamic Quality : {quality_history[-1]}'
        with open(f'final_configs{fake.bothify('############')}.txt','w',encoding='utf-8') as file:
            file.write(string + '\n')
            file.close()
    else:
        dpg.set_value('warning', value=' [You cannot save configurations] ')

def delete_selected_body():
    global emitted_function , u , v , p , texture_data , mask , is_running , is_paused
    ny , nx = mask.shape
    u = np.zeros((ny, nx), dtype=np.float32)
    v = np.zeros((ny, nx), dtype=np.float32)
    p = np.zeros((ny, nx), dtype=np.float32)
    texture_data = np.zeros(ny * nx * 4, dtype=np.float32)
    dpg.set_value('series_cp_upper', [[], []])
    dpg.set_value('series_cp_lower', [[], []])
    dpg.set_value('series_cl_lift', [[], []])
    dpg.set_value('series_cd_drag', [[], []])
    dpg.fit_axis_data("X_axis_pressure")
    dpg.fit_axis_data("Y_axis_pressure")
    dpg.fit_axis_data("X_axis_lift")
    dpg.fit_axis_data("Y_axis_lift")
    dpg.fit_axis_data("X_axis_drag")
    dpg.fit_axis_data("Y_axis_drag")
    mask = np.zeros((ny,nx),dtype=bool)
    dpg.set_value('fluid_texture',texture_data)
    is_running = False
    emitted_function = True
    is_paused = False

def render_cover():
    with dpg.texture_registry():
        dpg.add_dynamic_texture(width=nx,height=ny,default_value=list(np.zeros(4*ny*nx)),tag='fluid_texture')
    with dpg.file_dialog(directory_selector=False,show=False,callback=image_callback_data,id='dialog_window',width=500,height=500):
        dpg.add_file_extension(".png",color=(246, 157, 57),custom_text="[Image]")
        dpg.add_file_extension(".jpg",color=(246, 157, 57), custom_text="[Image]")
        dpg.add_file_extension(".jpeg",color=(246, 157, 57), custom_text="[Image]")
    with dpg.window(label='main tab', width=screen_width, height=screen_height, tag='main tab', no_resize=True,
                    no_move=True, no_collapse=True,show=True,no_title_bar=True):
        with dpg.group(horizontal=True):
            with dpg.plot(label="Pressure distribution",height=350,width=(screen_width // 3) * 0.98):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis,label='step',tag='X_axis_pressure')
                dpg.add_plot_axis(dpg.mvYAxis,label='pressure distribution',tag='Y_axis_pressure')
                dpg.add_line_series([],[],label='Upper surface',parent='Y_axis_pressure',tag='series_cp_upper')
                dpg.add_line_series([],[],label='Lower surface',parent='Y_axis_pressure',tag='series_cp_lower')
            with dpg.plot(label='Lift distribution',height=350,width=(screen_width // 3) * 0.98):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis,label='frames',tag='X_axis_lift')
                dpg.add_plot_axis(dpg.mvYAxis,label='Lift distribution',tag='Y_axis_lift')
                dpg.add_line_series([],[],label='Lift coefficient',parent='Y_axis_lift',tag='series_cl_lift')
            with dpg.plot(label='Drag distribution',height=350,width=(screen_width // 3) * 0.98):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis,label='frames',tag='X_axis_drag')
                dpg.add_plot_axis(dpg.mvYAxis,label='Drag distribution',tag='Y_axis_drag')
                dpg.add_line_series([],[],label='Drag coefficient',parent='Y_axis_drag',tag='series_cd_drag')
        dpg.add_spacer(width=2)
        with dpg.group(horizontal=True):
            with dpg.group(horizontal=False):
                dpg.add_spacer(width=2)
                dpg.add_slider_float(label='Input velocity(m/s)',default_value=50,width=int((screen_width // 3) * 0.98 * 2) + 7,min_value=10,max_value=500,tag='velocity_configuration')
                dpg.add_slider_float(label='Viscosity', default_value=1.5e-4,width=int((screen_width // 3) * 0.98 * 2) + 7,min_value=1e-6,max_value=1e-3,tag='viscosity_configuration',format="%.8f")
                dpg.add_slider_float(label='Mesh speed(horizontal)',default_value = 1,width=int((screen_width // 3) * 0.98 * 2) + 7,min_value = 0.1,max_value=10,tag='speed_mesh_X_configuration')
                dpg.add_slider_float(label='Mesh speed(vertical)',default_value=0,width=int((screen_width // 3) * 0.98 * 2) + 7,min_value=0.1,max_value=10,tag='speed_mesh_Y_configuration')
                dpg.add_slider_float(label='Angle of attack',default_value=0,width=int((screen_width // 3) * 0.98 * 2) + 7,min_value=-20.0,max_value=20.0,tag='angle_of_attack_configuration')
            with dpg.group(horizontal=False):
                with dpg.group(horizontal=True):
                    dpg.add_button(label='Choose subject', tag='update_configuration', callback=open_curtains,width=167,height=35)
                    dpg.add_button(label='Apply configurations',callback=update_configurations,width=167,height=35)
                with dpg.group(horizontal=True):
                    dpg.add_button(label='Pause simulation', tag='animation_view', callback=stop_animation,width=167,height=35)
                    dpg.add_button(label='Delete selected body',callback=delete_selected_body,width=167,height=35)
                with dpg.group(horizontal=True):
                    dpg.add_button(label='Save as CSV file',callback=save_as_csv,width=167,height=35)
                    dpg.add_button(label='Save as TXT file',callback=save_as_txt,width=167,height=35)
        with dpg.group(horizontal=True):
            with dpg.theme() as warning_theme:
                with dpg.theme_component(dpg.mvText):
                    dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 50, 50, 255))
            with dpg.group(horizontal=False):
                dpg.add_text('Aerodynamic Quality: 0', tag='current_aerodynamic_quality')
                dpg.add_image("fluid_texture", width= int((screen_width // 3) * 2 * 0.98 + 7), height=250)
            with dpg.plot(label='CPU usage',height=280,width=(screen_width // 3) * 0.98):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis,label='frames',tag='frames_for_CPU')
                dpg.add_plot_axis(dpg.mvYAxis,label='CPU load(%)',tag='CPU_check')
                dpg.add_line_series([],[],label='CPU load', parent='CPU_check', tag='series_CPU')
        with dpg.group(horizontal=True):
            dpg.add_text('It is not recommended to change default meshes speed otherwise flow might behave strange')
            dpg.add_text('', tag='warning')
        dpg.bind_item_theme('warning', warning_theme)

        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg,(129, 154, 145))
                dpg.add_theme_color(dpg.mvThemeCol_Text,(238, 239, 224))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,(247, 244, 234))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (209, 216, 190))
                dpg.add_theme_color(dpg.mvThemeCol_Button,(167, 193, 168))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (209,216,190))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,(238, 239, 224))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive,(238, 239, 224))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,(167,193,168))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (167,193,168))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive,(167,193,168))
                dpg.add_theme_color(dpg.mvPlotCol_PlotBg, (129, 154, 145),category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_PlotBorder, (209,216,190),category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_FrameBg,(167, 193, 168),category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_AxisGrid,(238, 239, 224),category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_LegendBg, (167, 193, 168),category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_LegendBorder,(238, 239, 224),category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_LegendText,(238, 239, 224),category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg,(167, 193, 168))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (238, 239, 224))
                dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg,(167,193,168))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,(167,193,168))
                dpg.add_theme_color(dpg.mvThemeCol_Header,(209, 216, 190))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive,(224, 195, 117))

        with dpg.theme() as lines_theme1:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line,(217, 34, 67),category=dpg.mvThemeCat_Plots)
        with dpg.theme() as lines_theme2:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line,(246, 157, 57),category=dpg.mvThemeCat_Plots)
        with dpg.theme() as lines_theme3:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line,(224, 195, 117),category=dpg.mvThemeCat_Plots)
        with dpg.theme() as lines_theme4:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line,(255, 245, 229),category=dpg.mvThemeCat_Plots)
        dpg.bind_theme(global_theme)
        dpg.bind_item_theme('series_cp_upper',lines_theme1)
        dpg.bind_item_theme('series_cp_lower',lines_theme2)
        dpg.bind_item_theme('series_cl_lift',lines_theme3)
        dpg.bind_item_theme('series_cd_drag',lines_theme4)
render_cover()
history_cl = []
history_cd = []
history_fps = []
fps_counter = 0
dpg.create_viewport(title='Flow over 2D subjects',width=450,height=250)
dpg.setup_dearpygui()
dpg.show_viewport()
aerodynamic_quality = 0
quality_history = []
time_history = []
cpu_usage_history = []
import psutil
psutil.cpu_percent(interval=None)
rows , cols = mask.shape
while dpg.is_dearpygui_running():
    if is_running and not is_paused:
        if emitted_function:
            history_cl.clear()
            history_cd.clear()
            history_fps.clear()
            time_history.clear()
            quality_history.clear()
            fps_counter = 0
            x_coords = []
            cp_upper = []
            cp_lower = []
            cp_front = []
            cp_back = []
            emitted_function = False
        result_velocity = np.mean(np.sqrt(u ** 2 + v ** 2))
        rho = 1.3
        p_inf = 0.0
        dynamic_pressure = 0.5 * rho * (u_uniform ** 2)
        u,v,p,new_rgba = computational_system(mask,u,v,p,texture_data,u_uniform,viscosity,u_mesh,v_mesh,nx,ny)
        dpg.set_value('fluid_texture',new_rgba)
        np.mean(np.sqrt(u**2+v**2))
        x_coords = []
        cp_upper = []
        cp_lower = []
        cp_front = []
        cp_back = []

        if fps_counter > 7250:
            is_paused = True

        for j in range(1,cols-1):
            wing_indices_j = np.where(mask[:, j])[0]
            if len(wing_indices_j):
                x_coords.append(j * dx)
                upper_wall_idx = wing_indices_j[0]
                lower_wall_idx = wing_indices_j[-1]
                if (upper_wall_idx - 2 < 0) or (lower_wall_idx + 2 > cols):
                    continue
                p_top = p[upper_wall_idx - 2, j]
                p_bottom = p[lower_wall_idx + 2, j]
                cp_t = (p_top - p_inf) / dynamic_pressure
                cp_b = (p_bottom - p_inf) / dynamic_pressure
                cp_upper.append(cp_t)
                cp_lower.append(cp_b)

        if len(x_coords) > 0:
            dpg.set_value('series_cp_upper',[x_coords,cp_upper])
            dpg.set_value('series_cp_lower',[x_coords,cp_lower])
            dpg.fit_axis_data('X_axis_pressure')
            dpg.fit_axis_data('Y_axis_pressure')

        for i in range(1,rows-1):
            wing_indices_i = np.where(mask[i,:])[0]
            if len(wing_indices_i):
                front_wall_idx = wing_indices_i[0]
                back_wall_idx = wing_indices_i[-1]
                if (front_wall_idx - 2 < 0) or (back_wall_idx + 2 > cols):
                    continue
                p_front = p[i,front_wall_idx - 2]
                p_back = p[i,back_wall_idx + 2]
                cp_b = (p_back - p_inf) / dynamic_pressure
                cp_f = (p_front - p_inf) / dynamic_pressure
                cp_back.append(cp_b)
                cp_front.append(cp_f)

        if len(cp_front) > 0 and len(cp_back) > 0 and len(cp_upper) > 0 and len(cp_lower) > 0:
            chord_length = len(cp_upper) * dx
            cp_diff = np.array(cp_lower) - np.array(cp_upper)
            cp_diff_drag = np.array(cp_front) - np.array(cp_back)
            c_lift = np.sum(cp_diff) / chord_length
            c_drag = (np.sum(cp_diff_drag) / chord_length)
            history_cl.append(c_lift)
            history_cd.append(c_drag)
            history_fps.append(fps_counter)
            if c_drag == 0:
                aerodynamic_quality = 0.0
            else:
                aerodynamic_quality = round(c_lift / c_drag, 3)
            quality_history.append(aerodynamic_quality)

        cpu_value = psutil.cpu_percent(interval=None)
        cpu_usage_history.append(cpu_value)
        dpg.set_value('series_CPU',[history_fps,cpu_usage_history])
        dpg.set_value('series_cl_lift',[history_fps,history_cl])
        dpg.set_value('series_cd_drag',[history_fps,history_cd])
        dpg.fit_axis_data('X_axis_lift')
        dpg.fit_axis_data('Y_axis_lift')
        dpg.fit_axis_data('X_axis_drag')
        dpg.fit_axis_data('Y_axis_drag')
        dpg.fit_axis_data('frames_for_CPU')
        dpg.fit_axis_data('CPU_check')
        dpg.set_value('current_aerodynamic_quality',f'Aerodynamic Quality: {aerodynamic_quality}')
        fps_counter += 1
        time_history.append(time.perf_counter() - start_time)
    dpg.render_dearpygui_frame()
dpg.destroy_context()









