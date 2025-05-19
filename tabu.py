import tkinter as tk
from tkinter import ttk, messagebox
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TabuSearchPlanner:
    def __init__(self, master):
        self.master = master
        self.master.title("Smart Travel Route Planner")
        self.locations = []
        self.weights = []
        self.penalty_weight = tk.DoubleVar(value=0.5)
        self.coords = []  # Untuk menyimpan koordinat visualisasi
        self.setup_ui()
        self.tabu_process = None
        
        # Variabel untuk tracking optimum solution
        self.no_improvement_count = 0
        self.max_no_improvement = 15  # Berhenti jika tidak ada peningkatan setelah 15 iterasi

    def setup_ui(self):
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Input Frame
        input_frame = tk.LabelFrame(main_frame, text="Input Lokasi", padx=5, pady=5)
        input_frame.pack(fill=tk.X, pady=5)

        tk.Label(input_frame, text="Nama Lokasi:").grid(row=0, column=0, padx=5)
        self.entry_name = tk.Entry(input_frame, width=20)
        self.entry_name.grid(row=0, column=1, padx=5)

        tk.Label(input_frame, text="Bobot Biaya:").grid(row=0, column=2, padx=5)
        self.entry_weight = tk.Entry(input_frame, width=10)
        self.entry_weight.insert(0, "1.0")
        self.entry_weight.grid(row=0, column=3, padx=5)

        tk.Button(input_frame, text="Add", command=self.add_location, width=8).grid(row=0, column=4, padx=5)

        tk.Label(input_frame, text="Denda per Unit (>300km):").grid(row=1, column=0, padx=5)
        tk.Entry(input_frame, textvariable=self.penalty_weight, width=10).grid(row=1, column=1, padx=5, sticky='w')

        # Locations Table Frame
        table_frame = tk.LabelFrame(main_frame, text="Daftar Lokasi", padx=5, pady=5)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        columns = ('no', 'lokasi', 'bobot', 'biaya_per_langkah', 'total_biaya')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        
        # Configure columns
        self.tree.heading('no', text='No')
        self.tree.heading('lokasi', text='Lokasi')
        self.tree.heading('bobot', text='Bobot')
        self.tree.heading('biaya_per_langkah', text='Biaya per Langkah')
        self.tree.heading('total_biaya', text='Total Biaya')
        
        self.tree.column('no', width=50, anchor='center')
        self.tree.column('lokasi', width=150)
        self.tree.column('bobot', width=80, anchor='center')
        self.tree.column('biaya_per_langkah', width=120, anchor='center')
        self.tree.column('total_biaya', width=100, anchor='center')

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Button Frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)

        tk.Button(button_frame, text="Delete Selected", command=self.delete_selected, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Run Tabu Search", command=self.run_tabu_search, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Reset All", command=self.reset_all, width=15).pack(side=tk.RIGHT, padx=5)

        # Best Solution Frame - NEW
        best_solution_frame = tk.LabelFrame(main_frame, text="Solusi Terbaik Saat Ini", padx=5, pady=5)
        best_solution_frame.pack(fill=tk.X, pady=5)
        
        self.best_solution_text = tk.Text(best_solution_frame, height=3, width=80, wrap=tk.WORD)
        self.best_solution_text.pack(fill=tk.X, padx=2, pady=2)
        self.best_solution_text.config(state=tk.DISABLED, bg='#f0f0f0')
        
        # Process Table Frame
        process_frame = tk.LabelFrame(main_frame, text="Proses Tabu Search", padx=5, pady=5)
        process_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        columns = ('iter', 'rute', 'total_biaya', 'aksi', 'status')
        self.process_tree = ttk.Treeview(process_frame, columns=columns, show='headings')
        
        self.process_tree.heading('iter', text='Iterasi')
        self.process_tree.heading('rute', text='Rute Saat Ini')
        self.process_tree.heading('total_biaya', text='Total Biaya')
        self.process_tree.heading('aksi', text='Aksi Tabu')
        self.process_tree.heading('status', text='Status')
        
        self.process_tree.column('iter', width=60, anchor='center')
        self.process_tree.column('rute', width=250)
        self.process_tree.column('total_biaya', width=100, anchor='center')
        self.process_tree.column('aksi', width=150)
        self.process_tree.column('status', width=100, anchor='center')

        process_scrollbar = ttk.Scrollbar(process_frame, orient="vertical", command=self.process_tree.yview)
        self.process_tree.configure(yscrollcommand=process_scrollbar.set)

        self.process_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        process_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Result and Visualization Frame
        result_frame = tk.Frame(main_frame)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.result_label = tk.Label(result_frame, text="", fg="green", wraplength=500, justify="left")
        self.result_label.pack(fill=tk.X, pady=5)

        self.canvas_frame = tk.Frame(result_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Initialize Tabu Search parameters
        self.iterations = 50
        self.tabu_size = 7
        self.current_iter = 0
        self.best_solution = None
        self.current_solution = None
        self.tabu_list = []
        self.dist_matrix = []

    def update_best_solution_display(self):
        """Update the best solution display with current best solution"""
        if self.best_solution is None:
            return
            
        self.best_solution_text.config(state=tk.NORMAL)
        self.best_solution_text.delete(1.0, tk.END)
        
        named_route = " → ".join([self.locations[i] for i in self.best_solution])
        best_cost, best_distance, best_penalty = self.calculate_route_cost(self.best_solution)
        
        best_solution_info = (f"Rute: {named_route}\n"
                             f"Total Jarak: {best_distance:.2f} km | "
                             f"Total Denda: {best_penalty:.2f} | "
                             f"Total Biaya: {best_cost:.2f}")
                             
        self.best_solution_text.insert(tk.END, best_solution_info)
        self.best_solution_text.config(state=tk.DISABLED)
        
        # Highlight the text box to make it more visible when updated
        self.best_solution_text.config(bg='#e6ffe6')  # Light green background
        self.master.after(500, lambda: self.best_solution_text.config(bg='#f0f0f0'))  # Reset after 500ms

    def add_location(self):
        name = self.entry_name.get().strip()
        try:
            weight = float(self.entry_weight.get())
            if weight <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Bobot harus angka positif.")
            return

        if not name:
            messagebox.showerror("Error", "Nama lokasi tidak boleh kosong.")
            return

        self.tree.insert('', tk.END, values=(
            len(self.tree.get_children())+1, 
            name, 
            f"{weight:.2f}", 
            "-", 
            "0.00"
        ))
        self.entry_name.delete(0, tk.END)
        self.entry_weight.delete(0, tk.END)
        self.entry_weight.insert(0, "1.0")

    def delete_selected(self):
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Tidak ada lokasi yang dipilih.")
            return

        for item in selected_items:
            self.tree.delete(item)

        # Update numbering
        for i, item in enumerate(self.tree.get_children()):
            self.tree.set(item, column='no', value=i+1)

    def reset_all(self):
        self.tree.delete(*self.tree.get_children())
        self.process_tree.delete(*self.process_tree.get_children())
        self.result_label.config(text="")
        
        # Reset best solution display
        self.best_solution_text.config(state=tk.NORMAL)
        self.best_solution_text.delete(1.0, tk.END)
        self.best_solution_text.config(state=tk.DISABLED)
        
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        self.locations = []
        self.weights = []
        self.dist_matrix = []
        self.tabu_list = []
        self.current_iter = 0
        self.coords = []
        self.no_improvement_count = 0
        self.best_solution = None

    def run_tabu_search(self):
        items = self.tree.get_children()
        if len(items) < 2:
            messagebox.showerror("Error", "Minimal dua lokasi diperlukan.")
            return

        # Clear previous results
        self.process_tree.delete(*self.process_tree.get_children())
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        self.result_label.config(text="")
        
        # Reset best solution display
        self.best_solution_text.config(state=tk.NORMAL)
        self.best_solution_text.delete(1.0, tk.END)
        self.best_solution_text.config(state=tk.DISABLED)

        # Get locations and weights
        self.locations = []
        self.weights = []
        for item in items:
            values = self.tree.item(item)['values']
            self.locations.append(values[1])  # Lokasi
            self.weights.append(float(values[2]))  # Bobot

        try:
            penalty = float(self.penalty_weight.get())
            if penalty < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Denda harus berupa angka positif.")
            return

        # Generate distance matrix and coordinates for visualization
        self.dist_matrix = self.generate_mock_distances(len(self.locations))
        self.coords = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(len(self.locations))]
        self.penalty = penalty

        # Initialize Tabu Search
        self.current_iter = 0
        self.current_solution = list(range(len(self.locations)))
        random.shuffle(self.current_solution)
        self.best_solution = self.current_solution[:]
        self.best_cost, self.best_distance, self.best_penalty = self.calculate_route_cost(
            self.best_solution)
        self.tabu_list = []
        self.no_improvement_count = 0
        
        # Update best solution display
        self.update_best_solution_display()

        # Add initial solution to process table
        self.add_to_process_table(
            iteration=0,
            route=self.current_solution,
            cost=self.best_cost,
            action="Inisialisasi",
            status="Solusi Awal"
        )

        # Start the search
        self.run_next_step()

    def run_next_step(self):
        if self.current_iter >= self.iterations or self.no_improvement_count >= self.max_no_improvement:
            self.finish_search()
            return

        # Generate neighbors not in tabu list
        all_neighbors = self.get_neighbors(self.current_solution)
        
        # Evaluate all neighbors
        neighbor_costs = []
        for neighbor in all_neighbors:
            cost, _, _ = self.calculate_route_cost(neighbor)
            is_tabu = neighbor in self.tabu_list
            neighbor_costs.append((neighbor, cost, is_tabu))
        
        # Sort by cost
        neighbor_costs.sort(key=lambda x: x[1])
        
        # Find best non-tabu neighbor
        best_neighbor = None
        best_neighbor_cost = float('inf')
        tabu_neighbors = []
        
        for neighbor, cost, is_tabu in neighbor_costs:
            if not is_tabu:
                if cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = cost
                    break  # Take first best non-tabu
            else:
                tabu_neighbors.append((neighbor, cost))

        # Use aspiration criteria if all neighbors are tabu
        if best_neighbor is None and tabu_neighbors:
            # Choose best tabu solution as aspiration
            best_neighbor, best_neighbor_cost = min(tabu_neighbors, key=lambda x: x[1])
            aspiration_used = True
        else:
            aspiration_used = False

        # Update current solution
        if best_neighbor:
            self.current_solution = best_neighbor[:]
            current_cost, current_distance, current_penalty = self.calculate_route_cost(self.current_solution)
            
            # Update best solution if improved
            if current_cost < self.best_cost:
                self.best_solution = self.current_solution[:]
                self.best_cost = current_cost
                self.best_distance = current_distance
                self.best_penalty = current_penalty
                status = "Lebih Baik! 🔄"
                self.no_improvement_count = 0
                
                # Update best solution display when a better solution is found
                self.update_best_solution_display()
            else:
                status = "Tidak Ada Peningkatan"
                self.no_improvement_count += 1

            # Update tabu list
            self.tabu_list.append(best_neighbor)
            if len(self.tabu_list) > self.tabu_size:
                self.tabu_list.pop(0)

            # Add to process table
            action_str = f"Tabu List: {len(self.tabu_list)}"
            if aspiration_used:
                action_str += " (Aspiration Used)"
                status += " 🔒 TABU"
                
            self.add_to_process_table(
                iteration=self.current_iter + 1,
                route=self.current_solution,
                cost=current_cost,
                action=action_str,
                status=status
            )

            # Update visualization
            self.plot_route(self.current_solution)
            
            self.current_iter += 1
            
            # Visual notification of optimal solution found
            if self.no_improvement_count >= self.max_no_improvement:
                self.result_label.config(
                    text=f"Solusi optimal ditemukan! Tidak ada peningkatan setelah {self.max_no_improvement} iterasi.",
                    fg="blue"
                )
            
            self.master.after(500, self.run_next_step)
        else:
            # No valid moves
            self.finish_search()

    def finish_search(self):
        # Update the locations table with costs
        self.update_costs_table()
        
        # Ensure best solution display is updated with final values
        self.update_best_solution_display()
        
        # Display final result
        named_route = " → ".join([self.locations[i] for i in self.best_solution])
        self.result_label.config(
            text=f"✓ Rute optimal ditemukan setelah {self.current_iter} iterasi:\n\n"
                 f"Rute Terbaik: {named_route}\n"
                 f"Total Jarak: {self.best_distance:.2f} km\n"
                 f"Total Denda: {self.best_penalty:.2f}\n"
                 f"Total Biaya: {self.best_cost:.2f}",
            fg="green"
        )

        # Plot the best route
        self.plot_route(self.best_solution, is_final=True)
        
        # Highlight the best solution box to draw attention
        self.best_solution_text.config(bg='#e6ffe6')  # Light green background
        self.master.after(2000, lambda: self.best_solution_text.config(bg='#f0f0f0'))  # Reset after 2 seconds

    def update_costs_table(self):
        """Update the costs in the locations table based on best solution"""
        for i, item in enumerate(self.tree.get_children()):
            loc_idx = self.best_solution[i]
            next_loc_idx = self.best_solution[(i + 1) % len(self.best_solution)]
            
            distance = self.dist_matrix[loc_idx][next_loc_idx]
            weight = self.weights[next_loc_idx]
            cost = distance * weight
            
            # Update biaya per langkah
            self.tree.set(item, column='biaya_per_langkah', value=f"{cost:.2f}")
            
            # Update total biaya (cumulative)
            if i == 0:
                total = cost
            else:
                prev_total = float(self.tree.item(self.tree.prev(item))['values'][4])
                total = prev_total + cost
            
            self.tree.set(item, column='total_biaya', value=f"{total:.2f}")

    def generate_mock_distances(self, n):
        """Generate symmetric distance matrix with random values"""
        dist = [[0 if i == j else random.randint(50, 200) for j in range(n)] for i in range(n)]
        # Make symmetric
        for i in range(n):
            for j in range(i+1, n):
                dist[j][i] = dist[i][j]
        return dist

    def calculate_route_cost(self, route):
        """Calculate total cost, distance and penalty for a route"""
        total_distance = 0
        total_cost = 0
        
        for i in range(len(route)):
            src = route[i]
            dst = route[(i + 1) % len(route)]
            dist = self.dist_matrix[src][dst]
            total_distance += dist
            total_cost += dist * self.weights[dst]
        
        penalty = max(0, total_distance - 300) * self.penalty
        total_cost += penalty
        
        return total_cost, total_distance, penalty

    def get_neighbors(self, solution):
        """Generate neighbor solutions by swapping two cities"""
        neighbors = []
        for i in range(len(solution)):
            for j in range(i+1, len(solution)):
                neighbor = solution[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
        return neighbors

    def add_to_process_table(self, iteration, route, cost, action, status):
        """Add a row to the process table"""
        route_str = " → ".join([self.locations[i] for i in route])
        item_id = self.process_tree.insert('', tk.END, values=(
            iteration,
            route_str,
            f"{cost:.2f}",
            action,
            status
        ))
        
        # Highlight row if containing TABU
        if "TABU" in status:
            self.process_tree.tag_configure('tabu', background='#ffcccb')
            self.process_tree.item(item_id, tags=('tabu',))
        
        # Auto-scroll to bottom
        self.process_tree.yview_moveto(1)

    def plot_route(self, route, is_final=False):
        """Visualize the current route"""
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(7, 5))
        
        # Get coordinates for the route
        x = [self.coords[i][0] for i in route]
        y = [self.coords[i][1] for i in route]
        labels = [self.locations[i] for i in route]

        # Plot the route
        line_color = 'green' if is_final else 'blue'
        ax.plot(x + [x[0]], y + [y[0]], '-', color=line_color, marker='o', linewidth=2)  # Return to start
        
        # Annotate points with location names
        for i, (xi, yi, label) in enumerate(zip(x, y, labels)):
            ax.annotate(label, (xi, yi), textcoords="offset points", 
                       xytext=(0, 10), ha='center', fontsize=9, weight='bold')
            
            # Add distance information
            if i < len(route) - 1:
                next_loc = route[i+1]
                dist = self.dist_matrix[route[i]][next_loc]
                ax.text((xi + x[i+1])/2, (yi + y[i+1])/2, 
                       f"{dist}km", ha='center', va='center', 
                       fontsize=8, color='red', bbox=dict(facecolor='white', alpha=0.7))
            elif i == len(route) - 1:
                # Distance back to start
                dist = self.dist_matrix[route[i]][route[0]]
                ax.text((xi + x[0])/2, (yi + y[0])/2, 
                       f"{dist}km", ha='center', va='center', 
                       fontsize=8, color='red', bbox=dict(facecolor='white', alpha=0.7))

        title_prefix = "Rute Optimal" if is_final else f"Visualisasi Rute (Iterasi {self.current_iter})"
        ax.set_title(title_prefix)
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def main():
    root = tk.Tk()
    root.geometry("1000x800")
    app = TabuSearchPlanner(root)
    root.mainloop()

if __name__ == "__main__":
    main()
Improve
Explain
