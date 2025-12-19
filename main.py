import matplotlib.pyplot as plt
import math
import copy
from dataclasses import dataclass, field
from typing import List, Tuple
from matplotlib.lines import Line2D


@dataclass
class Task:
    id: int
    c: float  # Tempo de Execução (WCET)
    t: float  # Período

    # Atributos calculados automaticamente
    utilization: float = field(init=False)

    # Estado da simulação
    next_release: float = 0.0
    abs_deadline: float = 0.0
    remaining_time: float = 0.0
    job_id: int = 0

    def __post_init__(self):
        self.utilization = self.c / self.t

    def reset(self):
        """Reinicia o estado da tarefa para uma nova simulação."""
        self.next_release = 0.0
        self.abs_deadline = 0.0
        self.remaining_time = 0.0
        self.job_id = 0


@dataclass
class Processor:
    id: int
    tasks: List[Task] = field(default_factory=list)
    # Lista de tuplas: (task_id, job_id, start, duration, is_completion)
    history: List[Tuple] = field(default_factory=list)

    @property
    def load(self):
        return sum(t.utilization for t in self.tasks)


class RTS_Simulator:
    def __init__(self, tasks: List[Task], m: int, algorithm: str):
        self.tasks = tasks
        self.m = m
        self.algorithm = algorithm
        self.processors = [Processor(i) for i in range(m)]
        self.success = False

    def _get_max_utilization(self, n: int) -> float:
        """Retorna o limite de utilização baseado no algoritmo e número de tarefas."""
        if self.algorithm == "FF-RM":
            return n * (math.pow(2, 1 / n) - 1)  # Limite Liu & Layland
        return 1.0  # EDF (100%)

    def run_partitioning(self) -> bool:
        # Ordena por período (T) para RM, mantém consistente para EDF
        queue = sorted(self.tasks, key=lambda x: x.t)

        for task in queue:
            assigned = False
            for proc in self.processors:
                # Verifica se cabe no processador atual
                new_n = len(proc.tasks) + 1
                new_load = proc.load + task.utilization

                if new_load <= self._get_max_utilization(new_n):
                    proc.tasks.append(task)
                    assigned = True
                    break

            if not assigned:
                print(f"[{self.algorithm}] Falha ao alocar Tarefa {task.id}")
                return False

        self.success = True
        return True

    def simulate(self, duration: int):
        if not self.success:
            return

        # Define a função de prioridade fora do loop (Otimização)
        # RM: Menor Período (t) | EDF: Menor Deadline Absoluto
        priority_key = (
            (lambda x: x.t) if self.algorithm == "FF-RM" else (lambda x: x.abs_deadline)
        )

        for proc in self.processors:
            if not proc.tasks:
                continue

            # Reinicia tarefas alocadas neste processador
            for t in proc.tasks:
                t.reset()

            time = 0.0
            while time < duration:
                # 1. Liberação de Jobs (Chegada)
                next_event = duration
                for t in proc.tasks:
                    if t.next_release <= time and t.remaining_time <= 0:
                        t.job_id += 1
                        t.remaining_time = t.c
                        t.abs_deadline = t.next_release + t.t
                        t.next_release += t.t
                    # Rastreia o próximo evento de liberação para avançar o tempo depois
                    next_event = min(next_event, t.next_release)

                # 2. Seleção (Escalonador)
                ready_queue = [t for t in proc.tasks if t.remaining_time > 0]
                target = min(ready_queue, key=priority_key) if ready_queue else None

                # 3. Avanço do Tempo
                if target:
                    # Executa até acabar a tarefa OU até a próxima liberação de qualquer tarefa
                    finish_time = time + target.remaining_time
                    dt = min(finish_time, next_event) - time

                    is_done = (time + dt) >= finish_time

                    if dt > 0:
                        proc.history.append(
                            (target.id, target.job_id, time, dt, is_done)
                        )
                        target.remaining_time -= dt
                        time += dt
                    else:
                        time = next_event  # Previne loops infinitos se dt for zero
                else:
                    # Ocioso (Idle)
                    time = min(next_event, duration)


def plot_results(sim_rm, sim_edf, duration, title="Resultados da Simulação"):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Adiciona o título global da figura
    fig.suptitle(title, fontsize=16, fontweight="bold")

    simulations = zip(axes, [sim_rm, sim_edf])
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for ax, sim in simulations:
        ax.set_title(f"Algoritmo: {sim.algorithm}")
        ax.set_ylabel("Processadores")
        ax.set_yticks([i * 10 + 5 for i in range(sim.m)])
        ax.set_yticklabels([f"Proc {i}" for i in range(sim.m)])
        ax.set_ylim(0, sim.m * 10)
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        if not sim.success:
            ax.text(
                0.5,
                0.5,
                "FALHA NO ESCALONAMENTO",
                transform=ax.transAxes,
                ha="center",
                color="red",
                weight="bold",
                bbox=dict(facecolor="white", edgecolor="red"),
            )
            continue

        for i, proc in enumerate(sim.processors):
            ax.axhline(
                i * 10 + 5, color="gray", alpha=0.1, linewidth=10
            )  # Fundo da linha

            for task_id, job_id, start, dt, done in proc.history:
                color = colors[task_id % len(colors)]

                # Barra de execução
                ax.broken_barh(
                    [(start, dt)],
                    (i * 10 + 2, 6),
                    facecolors=color,
                    edgecolors="black",
                    linewidth=0.5,
                )

                # Texto (T1.1)
                if dt > 0.5:
                    ax.text(
                        start + dt / 2,
                        i * 10 + 5,
                        f"T{task_id}.{job_id}",
                        ha="center",
                        va="center",
                        color="white",
                        weight="bold",
                        fontsize=8,
                    )

                # Seta de conclusão
                if done:
                    ax.plot(
                        start + dt,
                        i * 10 + 8.5,
                        marker="v",
                        color="black",
                        markersize=6,
                    )

    axes[1].set_xlabel("Tempo")
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            label="Job Finalizado",
            markerfacecolor="black",
            markersize=8,
        )
    ]
    fig.legend(handles=legend_elements, loc="upper right")

    # Ajusta o layout para deixar espaço para o título (rect=[left, bottom, right, top])
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def run_comparison(tasks_input, m=2, time=20, title="Simulação"):
    """Função auxiliar para rodar e comparar os dois algoritmos."""
    print(f"\n--- Iniciando: {title} ---")

    # FF-RM
    sim_rm = RTS_Simulator(copy.deepcopy(tasks_input), m, "FF-RM")
    if sim_rm.run_partitioning():
        sim_rm.simulate(time)

    # FF-EDF
    sim_edf = RTS_Simulator(copy.deepcopy(tasks_input), m, "FF-EDF")
    if sim_edf.run_partitioning():
        sim_edf.simulate(time)

    plot_results(sim_rm, sim_edf, time, title)


if __name__ == "__main__":
    # Exemplo 1: Slide (Sucesso em ambos)
    tasks_slide = [Task(1, 2, 4), Task(2, 3, 6), Task(3, 6, 20)]
    run_comparison(
        tasks_slide, m=2, time=20, title="Cenário 1: Carga Baixa (Sucesso em RM e EDF)"
    )

    # Exemplo 2: RM Falha, EDF Sucesso
    # RM falha pois o limite para 2 tarefas no mesmo proc é ~83%, e aqui temos 90% (4/10 + 5/10)
    tasks_diff = [Task(1, 4, 10), Task(2, 5, 10), Task(3, 5, 10)]
    run_comparison(
        tasks_diff,
        m=2,
        time=20,
        title="Cenário 2: Carga Limite (RM Falha vs EDF Sucesso)",
    )

    # Exemplo 3: Carga Máxima (100% em ambos os núcleos)
    # Cenário: 4 tarefas idênticas de 50% cada.
    # Total = 200% de carga (precisa de 2 CPUs cheias).
    # O FF-EDF vai conseguir (100% por core).
    # O FF-RM vai falhar (pois o limite matemático dele para 2 tarefas é ~83%).
    tasks_full_load = [
        Task(1, 2, 4),
        Task(2, 2, 4),
        Task(3, 2, 4),
        Task(4, 2, 4),
    ]

    run_comparison(
        tasks_full_load,
        m=2,
        time=20,
        title="Cenário 3: Carga Máxima 100% (RM Falha vs EDF Sucesso)",
    )
