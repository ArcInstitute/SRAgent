"""
Enum classes for organism definitions used across SRAgent.
"""

from enum import Enum


class OrganismEnum(Enum):
    """Organism sequenced"""

    # mammals
    HUMAN = "Homo sapiens"
    MOUSE = "Mus musculus"
    RAT = "Rattus norvegicus"
    MACAQUE = "Macaca mulatta"
    MARMOSET = "Callithrix jacchus"
    HORSE = "Equus caballus"
    DOG = "Canis lupus"
    BOVINE = "Bos taurus"
    SHEEP = "Ovis aries"
    PIG = "Sus scrofa"
    RABBIT = "Oryctolagus cuniculus"
    NAKED_MOLE_RAT = "Heterocephalus glaber"
    CHIMPANZEE = "Pan troglodytes"
    GORILLA = "Gorilla gorilla"
    CAT = "Felis catus"  # NEW
    BONOBO = "Pan paniscus"  # NEW
    GREEN_MONKEY = "Chlorocebus aethiops"  # NEW
    GRAY_SHORT_TAILED_OPPOSUM = "Monodelphis domestica"  # NEW
    GOAT = "Capra hircus"  # NEW
    ALPACA = "Vicugna pacos"  # NEW
    CHINCHILLA = "Chinchilla lanigera"  # NEW
    DOMESTIC_GUINEA_PIG = "Cavia porcellus"  # NEW
    GOLDEN_HAMSTER = "Mesocricetus auratus"  # NEW
    EURASIAN_HEDGEHOG = "Erinaceus europaeus"  # NEW
    AMERICAN_MINK = "Neovison vison"  # NEW
    SUNDA_PANGOLIN = "Manis javanica"  # NEW
    PLATYPUS = "Ornithorhynchus anatinus"  # NEW
    FERRET = "Mustela putorius"  # NEW
    NORTHERN_TREE_SHREW = "Tupaia belangeri"  # NEW
    # birds
    CHICKEN = "Gallus gallus"
    ZEBRAFINCH = "Taeniopygia guttata"  # NEW
    GOOSE = "Anser cygnoides"  # NEW
    DUCK = "Anas platyrhynchos"  # NEW
    # reptiles
    TURTLE = "Trachemys scripta"  # NEW
    # amphibians
    FROG = "Xenopus tropicalis"
    AXOLOTL = "Ambystoma mexicanum"  # NEW
    # fish
    ZEBRAFISH = "Danio rerio"
    SALMON = "Salmo salar"  # NEW
    STICKLEBACK = "Gasterosteus aculeatus"  # NEW
    # invertebrates
    FRUIT_FLY = "Drosophila melanogaster"
    ROUNDWORM = "Caenorhabditis elegans"
    MOSQUITO = "Anopheles gambiae"
    BLOOD_FLUKE = "Schistosoma mansoni"
    # plants
    THALE_CRESS = "Arabidopsis thaliana"
    RICE = "Oryza sativa"
    TOMATO = "Solanum lycopersicum"
    CORN = "Zea mays"
    # microorganisms
    METAGENOME = "metagenome"
    # other
    OTHER = "other"
