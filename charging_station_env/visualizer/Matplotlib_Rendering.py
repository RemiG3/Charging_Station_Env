from charging_station_env.visualizer import Rendering_Base
from charging_station_env.transition.Constants import Status, ChargingStatus
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


class Matplotlib_Rendering(Rendering_Base):
    def __init__(self):
        super().__init__()
        self.render_first = True
    
    def __call__(self, env, obs):
        tau = env.step_time/60
        N, T = env.number_of_chargers, env.timestep//env.step_time
        min_T = 32
        max_T = env.TIMESTEP_MAX//env.step_time
        offset_T = max(0, T-min_T)

        triangle_height = 1.2  # Height of the triangle
        triangle_base = .24  # Base length of the triangle

        price = obs[0:4]*env.energy["normalizers"]["price"]
        pv_production = obs[4:8]*env.energy["normalizers"]["renewable"]

        if self.render_first:
            plt.ion()
            self.fig = plt.figure(figsize=(15*int(round(1/tau)), env.number_of_chargers))
            self.gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 6], width_ratios=[1, 1])
            self.ax2 = self.fig.add_subplot(self.gs[0, 0])
            self.ax1 = self.fig.add_subplot(self.gs[0, 1])
            self.ax4 = self.fig.add_subplot(self.gs[1, 0])
            self.ax5 = self.fig.add_subplot(self.gs[1, 1])
            self.ax3 = self.fig.add_subplot(self.gs[2, :])
            self.render_first = False

            self.nb_requests_history = []
            self.nb_waiting_history = []
            self.nb_rejected_history = []

        # Electricity Price
        self.ax1.clear()
        self.ax1.set_xlabel('')
        self.ax1.set_ylabel('')
        plt.draw()
        
        self.ax1.plot([f'{(i)/int(round(1/tau)):.2f}' for i in range(offset_T, T+1)], env.energy["price"][offset_T:T+1], 'r-', label='Electricity Price')
        #self.ax1.plot([f'{(i)/int(round(1/tau)):.2f}' for i in range(T, min(T+4, max_T+3))], env.energy["price"][T:min(T+4, max_T+4)], 'r--', label='Electricity Price', alpha=.5)
        self.ax1.plot([f'{(i)/int(round(1/tau)):.2f}' for i in range(T, min(T+4, max_T+3))], price[0:min(T+4, max_T+4)-T], 'r--', label='Electricity Price', alpha=.5)
        self.ax1.set_ylabel('Price ($/kWh)')
        self.ax1.tick_params(axis='x', which='major', labelsize=4)
        self.ax1.tick_params(axis='y', which='major', labelsize=4)
        self.ax1.set_title('Electricity Price Evolution', fontsize=8)
        # self.ax1.legend(loc='upper left')

        # Energy management
        self.ax2.clear()
        self.ax2.set_xlabel('')
        self.ax2.set_ylabel('')
        plt.draw()

        #grid_history = np.array(env.action_controller.grid_final_evol[offset_T:T])
        grid_history = np.array(env.action_controller.history['grid_history'][offset_T:T])
        grid_limit = np.repeat([env.grid_limit * (env.step_time / 60)], len(grid_history))
        pv_prod = np.array(env.energy["renewable"][offset_T:T+1]) * (env.step_time / 60)
        #pv_prod_pred = np.array(env.energy["renewable"][T:min(T+4, max_T+4)]) * (env.step_time / 60)
        pv_prod_pred = np.array(pv_production[0:min(T+4, max_T+4)-T]) * (env.step_time / 60)
        self.ax2.plot([f'{(i)/int(round(1/tau)):.2f}' for i in range(offset_T, offset_T+len(grid_limit))], grid_limit, 'r-', label='Grid Limit')
        self.ax2.plot([f'{(i)/int(round(1/tau)):.2f}' for i in range(offset_T, offset_T+len(grid_history))], grid_history, 'y-', label='Grid Energy')
        self.ax2.plot([f'{(i)/int(round(1/tau)):.2f}' for i in range(offset_T, offset_T+len(pv_prod))], pv_prod, 'g-', label='PV Production')
        self.ax2.plot([f'{(i)/int(round(1/tau)):.2f}' for i in range(T, T+len(pv_prod_pred))], pv_prod_pred, 'g--', label='PV Predicted', alpha=.5)
        self.ax2.set_ylabel('Energy (kWh)')
        self.ax2.tick_params(axis='x', which='major', labelsize=4)
        self.ax2.tick_params(axis='y', which='major', labelsize=4)
        self.ax2.set_title('Energy Management Evolution', fontsize=8)
        self.ax2.legend(loc='lower left', prop={'size': 6})

        # EV Demands
        self.ax4.clear()
        self.ax4.set_xlabel('')
        self.ax4.set_ylabel('')
        plt.draw()

        if T > 0:
            nb_past_rejected = len([list(req.values()) for req in env.scenario[T-1].values() if(req['status'] == Status.REJECTED)])
            nb_past_waiting = len([list(req.values()) for req in env.scenario[T-1].values() if(req['status'] == Status.WAITING)])
            if(len(self.nb_rejected_history) > 0):
                self.nb_rejected_history[-1] = nb_past_rejected
            else:
                self.nb_rejected_history.append(nb_past_rejected)
            if(len(self.nb_requests_history) == 0):
                nb_past_requests = len([list(req.values()) for req in env.scenario[T-1].values() if((req['status'] == Status.ARRIVED) or (req['status'] == Status.REJECTED))])
                self.nb_requests_history.append(nb_past_requests)
            
            self.nb_waiting_history.append(nb_past_waiting)
            self.ax4.plot([f'{(i)/int(round(1/tau)):.2f}' for i in range(offset_T, offset_T+len(self.nb_waiting_history[offset_T:T]))], self.nb_waiting_history[offset_T:T], 'b-', label='Number of Waiting')
        else:
            self.nb_requests_history = []
            self.nb_waiting_history = []
            self.nb_rejected_history = []

        nb_current_requests = len([list(req.values()) for req in env.scenario[T].values() if((req['status'] == Status.ARRIVED) or (req['status'] == Status.REJECTED))])
        nb_current_rejected = len([list(req.values()) for req in env.scenario[T].values() if(req['status'] == Status.REJECTED)])
        self.nb_rejected_history.append(nb_current_rejected)
        self.nb_requests_history.append(nb_current_requests)
        self.ax4.plot([f'{(i)/int(round(1/tau)):.2f}' for i in range(offset_T, offset_T+len(self.nb_requests_history[offset_T:T+1]))], self.nb_requests_history[offset_T:T+1], 'y-', label='Number of Requests')
        self.ax4.plot([f'{(i)/int(round(1/tau)):.2f}' for i in range(offset_T, offset_T+len(self.nb_rejected_history[offset_T:T+1]))], self.nb_rejected_history[offset_T:T+1], 'r-', label='Number of Rejected')
        self.ax4.set_ylabel('Number of demands')
        self.ax4.tick_params(axis='x', which='major', labelsize=4)
        self.ax4.tick_params(axis='y', which='major', labelsize=4)
        self.ax4.set_title('EV Demands Evolution', fontsize=8)
        self.ax4.legend(loc='lower left', prop={'size': 6})
        self.ax4.text(0 + .01, 1 - .075, f"Total Accepted={round(sum(self.nb_requests_history) - sum(self.nb_rejected_history)):.0f}", ha='left', va='top', transform=self.ax4.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", ec="black", fc="white"))

        # Reward history
        self.ax5.clear()
        self.ax5.set_xlabel('')
        self.ax5.set_ylabel('')
        plt.draw()

        reward_history = np.array(env.action_controller.history['reward_history'][offset_T:T])
        self.ax5.plot([f'{(i)/int(round(1/tau)):.2f}' for i in range(offset_T, offset_T+len(reward_history))], reward_history, color='black', linestyle='-', label='Reward')
        self.ax5.text(0 + .01, 1 - .075, f"Total Reward={round(sum(env.action_controller.history['reward_history'])):.0f}", ha='left', va='top', transform=self.ax5.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", ec="black", fc="white"))
        self.ax5.set_ylabel('Reward')
        self.ax5.tick_params(axis='x', which='major', labelsize=4)
        self.ax5.tick_params(axis='y', which='major', labelsize=4)
        self.ax5.set_title('Reward Evolution', fontsize=8)
        #ax2.legend(loc='upper left')

        # Scenario
        self.ax3.clear()
        self.ax3.set_xlabel('')
        self.ax3.set_ylabel('')
        self.ax3.set_yticks([])
        self.ax3.set_yticklabels([])
        self.ax3.set_xticks([])
        self.ax3.set_xticklabels([])
        self.ax3.grid(visible=False)
        for label in self.ax3.get_yticklabels():
            label.clear()
        for label in self.ax3.get_xticklabels():
            label.clear()
        plt.draw()

        T += 1
        offset_T = max(0, T-min_T)

        self.ax3.grid(visible=True)
        self.ax3.set_yticks([i*env.number_of_chargers for i in range(1, N+1)])
        self.ax3.set_yticklabels([str(i+1) for i in range(N)])
        dx = -.05; dy = -.4
        offset = matplotlib.transforms.ScaledTranslation(dx, dy, self.fig.dpi_scale_trans)
        for label in self.ax3.yaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)

        self.ax3.set_xticks([i for i in range(min(min_T, T))])
        self.ax3.set_xticklabels([f'{(i)/int(round(1/tau)):.2f}' for i in range(offset_T, T)])
        dx = +.2; dy = -.05
        xoffset = matplotlib.transforms.ScaledTranslation(dx, dy, self.fig.dpi_scale_trans)
        for label in self.ax3.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + xoffset)


        colors = {'charging': 'yellow', 'not_charging': 'gray'}
        labels = {'charging': 'Charging', 'not_charging': 'Not Charging'}
        first_dict_label = {'charging': False, 'not_charging': False, 'waiting': False, 'departure': False, 'arrival': False}
        last_soc_per_vehicle = {}
        charger_vehicle_association = {}
        set_ev_nums_accepted = set()
        set_ev_nums_waiting = set()
        for t in range(offset_T, T):
            for req in env.scenario[t].values():
                if ((req['status'] == Status.ACCEPTED) or (req['status'] == Status.FINISHED)):
                    if req['num'] not in last_soc_per_vehicle:
                        last_soc_per_vehicle[req['num']] = req['current_soc']
                        charger_vehicle_association[req['num']] = req['charger']
                    else:
                        was_charged = (last_soc_per_vehicle[req['num']] != req['current_soc'])
                        charging_key = 'charging' if was_charged else 'not_charging'
                        if not first_dict_label[charging_key]:
                            first_dict_label[charging_key] = True
                            label = labels[charging_key]
                        else:
                            label = None
                        self.ax3.broken_barh([(t-offset_T-1, 1)], (charger_vehicle_association[req['num']]*env.number_of_chargers+2.75, 4), facecolors=colors[charging_key], alpha=.5, label=label)
                
                if (req['status'] == Status.ACCEPTED):
                    self.ax3.text(t+.15-offset_T, env.number_of_chargers*req['charger']+4, f"{int(round(req['current_soc']*100))}%", fontsize=10)
                    last_soc_per_vehicle[req['num']] = req['current_soc']
                    
                    if (req['num'] not in set_ev_nums_accepted):
                        set_ev_nums_accepted.add(req['num'])
                        first_acc_arr = None
                        for sub_t in range(t):
                            for sub_req in env.scenario[sub_t].values():
                                if ((req['arrival']-offset_T > 0) and (req['num'] == sub_req['num']) and (sub_req['status'] == Status.WAITING) and (req['num'] not in set_ev_nums_waiting)):
                                    set_ev_nums_waiting.add(req['num'])
                                    if not first_dict_label['waiting']:
                                        first_dict_label['waiting'] = True
                                        label = 'Waiting Arrival'
                                    else:
                                        label = None
                                    arrival_triangle = patches.Polygon([(req['arrival'] + .5 - offset_T, req['charger']*env.number_of_chargers + 6.75),
                                                                        (req['arrival'] + triangle_base/2 + .5 - offset_T, req['charger']*env.number_of_chargers + 6.75 + triangle_height),
                                                                        (req['arrival'] - triangle_base/2 + .5 - offset_T, req['charger']*env.number_of_chargers + 6.75 + triangle_height)],
                                                                        closed=True, color='blue', label=label)
                                    self.ax3.add_patch(arrival_triangle)
                                    self.ax3.text(req['arrival']+.35 - offset_T, req['charger']*env.number_of_chargers+8.25, f"{int(round(req['num']))}", fontsize=9)
                                if ((first_acc_arr is None) and (sub_req['status'] == Status.ACCEPTED) and (req['num'] == sub_req['num'])):
                                    first_acc_arr = sub_t
                        if(first_acc_arr is None):
                            first_acc_arr = t

                        if not first_dict_label['arrival']:
                            first_dict_label['arrival'] = True
                            label = 'Plugged In'
                        else:
                            label = None
                        arrival_triangle = patches.Polygon([(first_acc_arr + .5 - offset_T, req['charger']*env.number_of_chargers + 6.75),
                                                            (first_acc_arr + triangle_base/2 + .5 - offset_T, req['charger']*env.number_of_chargers + 6.75 + triangle_height),
                                                            (first_acc_arr - triangle_base/2 + .5 - offset_T, req['charger']*env.number_of_chargers + 6.75 + triangle_height)],
                                                            closed=True, color='green', label=label)
                        if not first_dict_label['departure']:
                            first_dict_label['departure'] = True
                            label = 'Departure'
                        else:
                            label = None
                        departure_triangle = patches.Polygon([(req['departure'] - offset_T, req['charger']*env.number_of_chargers + 2.5),
                                                            (req['departure'] + triangle_base/2 - offset_T, req['charger']*env.number_of_chargers - triangle_height + 2.5),
                                                            (req['departure'] - triangle_base/2 - offset_T, req['charger']*env.number_of_chargers - triangle_height + 2.5)],
                                                            closed=True, color='red', label=label)
                        
                        self.ax3.add_patch(arrival_triangle)
                        self.ax3.text(t+.35 - offset_T, req['charger']*env.number_of_chargers+8.25, f"{int(round(req['num']))}", fontsize=9)
                        self.ax3.add_patch(departure_triangle)
                        #self.ax3.text(req['departure']-.15, req['charger']*env.number_of_chargers+.1, f"{int(round(req['num']))}", fontsize=8)
                        

        self.ax3.set_xlabel('Timestep (hour)')
        self.ax3.set_ylabel('Charger')
        self.ax3.set_ylim(0, 10*N)
        self.ax3.set_xlim(0, min(min_T, T))
        if (sum(first_dict_label.values()) > 0):
            self.ax3.legend(loc='upper left', prop={'size': 8})
        
        plt.tight_layout(pad=.25)
        # plt.legend()
        plt.draw()
        if (env.timestep+env.step_time < env.TIMESTEP_MAX):
            plt.pause(.25)
        else:
            plt.pause(10.)


