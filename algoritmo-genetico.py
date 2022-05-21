from collections import namedtuple
from functools import partial
from random import choices, randint, randrange, random
from typing import List, Optional, Callable, Tuple

Genome = List[int]
Population = List[Genome]
PopulateFunc = Callable[[], Population]
FitnessFunc = Callable[[Genome], int]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]
PrinterFunc = Callable[[Population, int, FitnessFunc], None]

ChicoDelBarrio= namedtuple('ChicoDelBarrio', ['accesorio', 'color', 'rol', 'apodo'])

chicosDelBarrio = [
    ChicoDelBarrio(0, 3, 6, 9),
    ChicoDelBarrio(12, 15, 18, 21),
    ChicoDelBarrio(24, 27, 30, 33),
    ChicoDelBarrio(36, 39, 42, 45),
    ChicoDelBarrio(48, 51, 54, 57),
]

invalidos = ['101', '110','111']

#ACCESORIOS
gafas_de_sol='000' 
gorro_de_aviador='001' 
buzo_con_capucha='010' 
gorra ='011' 
camisa_manga_larga ='100' 

acc = {
    '000': 'gafas_de_sol',
    '001': 'gorro_de_aviador',
    '010': 'buzo_con_capucha',
    '011': 'gorra',
    '100': 'camisa_manga_larga',
    '101': 'invalido',
    '110': 'invalido',
    '111': 'invalido'
}

#COLORES
naranja ='000' 
verde ='001' 
azul ='010' 
celeste ='011' 
rojo ='100'

col = {
    '000': 'naranja',
    '001': 'verde',
    '010': 'azul',
    '011': 'celeste',
    '100': 'rojo',
    '101': 'invalido',
    '110': 'invalido',
    '111': 'invalido'
}

#ROLES
líder ='000' 
sorpresa ='001' 
soldado ='010' 
estratega ='011' 
técnico ='100' 

role = {
    '000': 'líder',
    '001': 'sorpresa',
    '010': 'soldado',
    '011': 'estratega',
    '100': 'técnico',
    '101': 'invalido',
    '110': 'invalido',
    '111': 'invalido'
}

#APODOS
Migue ='000' 
Abby ='001' 
Guero ='010' 
Kuki ='011' 
Memo ='100' 

apod = {
    '000': 'Migue',
    '001': 'Abby',
    '010': 'Guero',
    '011': 'Kuki',
    '100': 'Memo',
    '101': 'invalido',
    '110': 'invalido',
    '111': 'invalido'
}

def generate_genome(length: int) -> Genome:
    accesorios = ['000', '001', '010', '011', '100', '101', '110','111']
    colores = ['000', '001', '010', '011', '100', '101', '110','111']
    roles = ['000', '001', '010', '011', '100', '101', '110','111']
    apodos = ['000', '001', '010', '011', '100', '101', '110','111']
    genome: list[int] = []

    for _ in range(length):
        genome.extend(list(accesorios.pop(randint(0, len(accesorios) - 1))))
        
        genome.extend(colores.pop(randint(0, len(colores) - 1)))

        genome.extend(roles.pop(randint(0, len(roles) - 1)))

        genome.extend(apodos.pop(randint(0, len(apodos) - 1)))

    return genome 

def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]

def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(gene) for gene in population],
        k=2
    )

def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of same length")

    length = len(a)
    if length < 2:
        return a, b

    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]


def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else str(abs(int(genome[index]) - 1))
    return genome

#SPECIFIC FUNCTIONS
def get_genome_value(genome: Genome, position: int) -> str:
    return ''.join(genome[position: (position + 3)])

def numero_uno_lleva_el_color_rojo(genome: Genome) -> int:
    if get_genome_value(genome, chicosDelBarrio[0].color) == rojo:
        return 10
    else:
        return (-5)

def numero_cuatro_es_apodado_Guero(genome: Genome) -> int:
    if get_genome_value(genome, chicosDelBarrio[3].apodo) == Guero:
        return 10
    else:
        return (-5)

def quien_viste_de_verde_posee_un_número_mayor_a_quien_cumple_el_rol_de_técnico(genome: Genome) -> int: 
    verde_index=-1
    tecnico_index=-1

    for i, cdb in enumerate(chicosDelBarrio):
        if get_genome_value(genome, cdb.color) == verde:
            verde_index = i

        if get_genome_value(genome, cdb.rol) == técnico:
            tecnico_index = i

    if verde_index > tecnico_index:
        return 10
    else:
        return (-5)

def quien_es_apodada_Abby_lleva_gorra(genome: Genome) -> int:
    for i, cdb in enumerate(chicosDelBarrio):
        if get_genome_value(genome, cdb.apodo) == Abby and get_genome_value(genome, cdb.accesorio) == gorra:
            return 10

    return (-5)

def la_persona_con_rol_sorpresa_usa_el_color_verde(genome: Genome) -> int:
    for i, cdb in enumerate(chicosDelBarrio):
        if get_genome_value(genome, cdb.rol) == sorpresa and get_genome_value(genome, cdb.color) == verde:
            return 10

    return (-5)

def la_persona_con_el_número_más_alto_viste_de_azul(genome: Genome) -> int:
    if get_genome_value(genome, chicosDelBarrio[4].color) == azul:
        return 10
    else:
        return (-5)

def la_persona_con_rol_sorpresa_usa_el_color_verde(genome: Genome) -> int:
    for i, cdb in enumerate(chicosDelBarrio):
        if get_genome_value(genome, cdb.accesorio) == gorro_de_aviador and get_genome_value(genome, cdb.color) == celeste:
            return 10

    return (-5)

def una_persona_junto_al_número_3_tiene_rol_técnico(genome: Genome) -> int:
    if get_genome_value(genome, chicosDelBarrio[1].rol) == técnico or get_genome_value(genome, chicosDelBarrio[3].rol) == técnico:
        return 10
    else:
        return (-5)

def quien_viste_de_naranja_tiene_un_número_mayor_a_quien_usa_gorro_de_aviador(genome: Genome) -> int: 
    naranja_index=-1
    aviador_index=-1

    for i, cdb in enumerate(chicosDelBarrio):
        if get_genome_value(genome, cdb.color) == naranja:
            naranja_index = i

        if get_genome_value(genome, cdb.accesorio) == gorro_de_aviador:
            aviador_index = i

    if naranja_index > aviador_index:
        return 10
    else:
        return (-5)

def quien_viste_de_naranja_tiene_un_número_menor_a_tiene_el_rol_estratega(genome: Genome) -> int: 
    naranja_index=-1
    estratega_index=-1

    for i, cdb in enumerate(chicosDelBarrio):
        if get_genome_value(genome, cdb.color) == naranja:
            naranja_index = i

        if get_genome_value(genome, cdb.rol) == estratega:
            estratega_index = i

    if naranja_index < estratega_index:
        return 10
    else:
        return (-5)

def el_numero_con_rol_sorpresa_se_encuentra_junto_al_numero_con_rol_soldado(genome: Genome) -> int: 
    sorpresa_index=-1
    soldado_index=-1

    for i, cdb in enumerate(chicosDelBarrio):
        if get_genome_value(genome, cdb.rol) == sorpresa:
            sorpresa_index = i

        if get_genome_value(genome, cdb.rol) == soldado:
            soldado_index = i

    if abs(sorpresa_index - soldado_index) == 1:
        return 10
    else:
        return (-5)

def penalizaciones(genome: Genome) -> int:
    value = 0

    for i, cdb in enumerate(chicosDelBarrio):
        if get_genome_value(genome, cdb.accesorio) in invalidos:
            value -= 30

        if get_genome_value(genome, cdb.color) in invalidos:
            value -= 30

        if get_genome_value(genome, cdb.rol) in invalidos:
            value -= 30

        if get_genome_value(genome, cdb.apodo) in invalidos:
            value -= 30

    return value

def fitness(genome: Genome) -> int:
    value = 0

    value += numero_uno_lleva_el_color_rojo(genome)
    
    value += numero_cuatro_es_apodado_Guero(genome)

    value += quien_viste_de_verde_posee_un_número_mayor_a_quien_cumple_el_rol_de_técnico(genome)
    
    value += quien_es_apodada_Abby_lleva_gorra(genome)

    value += quien_es_apodada_Abby_lleva_gorra(genome)

    value += la_persona_con_el_número_más_alto_viste_de_azul(genome)

    value += una_persona_junto_al_número_3_tiene_rol_técnico(genome)

    value += quien_viste_de_naranja_tiene_un_número_mayor_a_quien_usa_gorro_de_aviador(genome)

    value += el_numero_con_rol_sorpresa_se_encuentra_junto_al_numero_con_rol_soldado(genome)

    value += penalizaciones(genome)
    
    return value
# 
# 
# 
# 
# 

def population_fitness(population: Population, fitness_func: FitnessFunc) -> int:
    return sum([fitness_func(genome) for genome in population])


def sort_population(population: Population, fitness_func: FitnessFunc) -> Population:
    return sorted(population, key=fitness_func, reverse=True)


def genome_to_string(genome: Genome) -> str:
    return "".join(map(str, genome))

def genome_to_knd(team: str) -> str:
    for i in range(0, len(team), 12):
        knd = team[i:i+12] 
        print("%i" % ((i/12) +1))
        print("%s" % acc[knd[0:3]])
        print("%s" % col[knd[3:6]])
        print("%s" % role[knd[6:9]])
        print("%s" % apod[knd[9:12]])


def print_stats(population: Population, generation_id: int, fitness_func: FitnessFunc):
    print("GENERATION %02d" % generation_id)
    print("=============")
    print("Population: [%s]" % ", ".join([genome_to_string(gene) for gene in population]))
    print("Avg. Fitness: %f" % (population_fitness(population, fitness_func) / len(population)))
    sorted_population = sort_population(population, fitness_func)
    print(
        "Best: %s (%f)" % (genome_to_string(sorted_population[0]), fitness_func(sorted_population[0])))
    genome_to_knd(genome_to_string(sorted_population[0]))
    print("Worst: %s (%f)" % (genome_to_string(sorted_population[-1]),
                              fitness_func(sorted_population[-1])))
    genome_to_knd(genome_to_string(sorted_population[-1]))
    print("")

    average = (population_fitness(population, fitness_func) / len(population))
    
    return average

def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        fitness_limit: int,
        number: int,
        probability: float,
        selection_func: SelectionFunc = selection_pair,
        crossover_func: CrossoverFunc = single_point_crossover,
        mutation_func: MutationFunc = mutation,
        generation_limit: int = 100,
        printer: Optional[PrinterFunc] = None) \
        -> Tuple[Population, int]:
        
    population = populate_func()
    averages: List[int] = []
    for i in range(generation_limit):
        population = sorted(population, key=lambda genome: fitness_func(genome), reverse=True)

        if printer is not None:
            averages.append(printer(population, i, fitness_func))

        if fitness_func(population[0]) >= fitness_limit:
            break

        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a, number, probability)
            offspring_b = mutation_func(offspring_b, number, probability)
            next_generation += [offspring_a, offspring_b]

        population = next_generation

    print("AVERAGES")
    for element in averages:
        print(str(element))

    return population, i

population, generations = run_evolution(
    populate_func=partial(
        generate_population, size=20, genome_length=len(chicosDelBarrio)
    ),
    number=1, 
    probability=0.5,
    fitness_func=fitness,
    fitness_limit=90,
    generation_limit=100,
    printer=print_stats
)