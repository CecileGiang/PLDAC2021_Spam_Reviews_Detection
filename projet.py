# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 10:57:05 2021

@author: GIANG Cécile, LENOIR Romain
"""

########################## IMPORTATION DES DONNEES ###########################

import json
import matplotlib.pyplot as plt
import numpy as np
import copy
from sklearn.neighbors.kde import KernelDensity
import seaborn as sns
from scipy.signal import find_peaks


######################## NON-AFFICHAGE DES WARNINGS  #########################

import warnings
warnings.filterwarnings("ignore")


########################### PREPARATION DES DONNEES ##########################

def parse(filename, data_number):
    """ Permet de parser un fichier de données Amazon et crée notre dataset.
        Renvoie la liste des échantillons de données par review et la liste 
        des attributs associés à ces échantillons.
        @param filename: str, nom du fichier à parser
        @data_number: int, nombre de reviews à prendre dans le fichier.
                      Prend les data_number premiers reviews
        @return data: list(str), liste des échantillons de données, chaque
                      échantillon correspondant à un avis posté.
        @return fields: list(str), liste des attributs
    """
    # ------- Ouverture du fichier, initialisation de data et fields
    file = open(filename, 'r')
    data, fields = [], []
    
    # ------- Mise à jour de data et fields par parcours de fichier
    
    for i in range(data_number):
        
        # items = dictionnaire associé à la ligne lue line
        line = file.readline()
        items = json.loads(line)
        
        # ------------------------- MAJ de fields -------------------------
        
        # keys: liste des clés du dictionnaire associé à la ligne courante
        keys = list( items.keys() )
        
        # Cas où la ligne courante présente de nouveaux attributs
        if keys != fields:
            
            new_keys = [k for k in keys if k not in fields]
            fields += new_keys
            
            # Mise à jour des données déjà lues avec les nouvelles clés
            for j in range(i):
                data[j] += [None]*len(new_keys)
                
        # -------------------------- MAJ de data --------------------------
        
        data.append([None]*len(fields))
        
        for index_label in range(len(fields)):
            if fields[index_label] in items.keys():
                data[i][index_label] = items[fields[index_label]]
        
    return data, fields

## Création du dataset
#data, fields = parse('data/cellphones_accessories.json', data_number=10000)

def count(filename):
    file = open(filename, 'r')
    count = {}
    
    line = file.readline()
    items = json.loads(line)
    
    while line:
        count[items['asin']] = count.get(items['asin'], 0) + 1
    
    return count
    
def parse_some(to_read, to_save, nprods = 500):
    """ Permet de parser un fichier de données Amazon et crée notre dataset.
        Contrairement à la fonction précédente, lit l'intégralité du fichier 
        et ne sauvegarde que les revues associées aux nprods premiers produits.
        Renvoie la liste des échantillons de données par review et la liste 
        des attributs associés à ces échantillons.
        @param to_read: str, nom du fichier à parser
        @param to_save: str, nom du fichier dans lequel stocker les données
        @nprods: int, nombre de produits à prendre dans le fichier.
        @return data: list(str), liste des échantillons de données, chaque
                      échantillon correspondant à un avis posté.
        @return fields: list(str), liste des attributs
    """
    # ------- Ouverture du fichier, initialisation de data et fields
    fread = open(to_read, 'r')
    fsave = open(to_save, 'w')
    data, fields = [], []
    
    # ------- On maintient la liste des produits rencontrés
    list_prods = set()
    
    # ------- Mise à jour de data et fields par parcours de fichier
    line = fread.readline()
    items = json.loads(line)
    
    i = 0
    
    while line:
        # On rajoute la donnée dans le dataset si on n'a pas atteint nprods produits
        # ou si c'est l'un des produits à garder
        if len(list_prods) < nprods or items['asin'] in list_prods:
            
            # ------------------------- MAJ de fields -------------------------
            
            list_prods.add( items['asin'] )
            fsave.write(line)
            
            # ------------------------- MAJ de fields -------------------------
        
            # keys: liste des clés du dictionnaire associé à la ligne courante
            keys = list( items.keys() )
            
            # Cas où la ligne courante présente de nouveaux attributs
            if keys != fields:
                
                new_keys = [k for k in keys if k not in fields]
                fields += new_keys
                
                # Mise à jour des données déjà lues avec les nouvelles clés
                for j in range(i):
                    data[j] += [None]*len(new_keys)
                    
            # -------------------------- MAJ de data --------------------------
            
            data.append([None]*len(fields))
            
            for index_label in range(len(fields)):
                if fields[index_label] in items.keys():
                    data[i][index_label] = items[fields[index_label]]
            
            i += 1
        
        line = fread.readline()
        if line:
            items = json.loads(line)
    
    # On remplace les textes de reviews vides par ''
    index_revs = fields.index('reviewText')
    
    for i in range(len(data)):
        if data[i][index_revs]==None:
            data[i][index_revs] = ''
    
    fread.close()
    fsave.close()
        
    return data, fields


####################### DETECTION DE BURST DE REVIEWS ########################

# Détection de bursts de reviews, qui indiquent des périodes de temps pendant
# lesquelles le nombre de reviews postés est anormalement élevé, et où l'on
# peut suspecter la présence de reviews spam

def reviewPerMonth(data, fields):
    """ Renvoie la liste du nombre de reviews posté par mois dans data.
        @param data: list(str), liste des échantillons de données, chaque
                      échantillon correspondant à un avis posté.
        @param fields: list(str), liste des attributs
        @return X: list(int), liste du nombre de reviews par mois
    """
    index_time = fields.index('reviewTime')
    
    # Temps sous la forme 'mois jour, annee'
    times = [d[index_time] for d in data]
    
    # Normalisation sous la forme (annee, mois, jour)
    times_normalized = []
    for t in times:
        mois, jour, annee = t.split()
        times_normalized.append((int(annee), int(mois), int(jour.strip(','))))
    
    # Tri des temps du plus ancien au plus récent
    times_normalized = sorted(times_normalized)
    
    # Plot le nombre de review par mois
    liste_annees = sorted( np.unique( [t[0] for t in times_normalized] ) )
    liste_mois = sorted( np.unique( [t[1] for t in times_normalized] ) )
    liste_jours = sorted( np.unique( [t[2] for t in times_normalized] ) )
    
    # X: liste du nombre de reviews postés par mois
    X = [0] * (len(liste_annees) * len(liste_mois))
    
    for annee, mois, jour in times_normalized:
        X[liste_annees.index(annee)*len(liste_mois) + liste_mois.index(mois)] += 1
    
    return X


# ------- Détection de bursts de reviews par moyenne mobile

def moyenneMobile(X, width):
    """ Calcule la moyenne mobile de la liste X sur une fenêtre de largeur width.
        @param X: list(int), liste du nombre de reviews par intervalle de temps
                  que l'on veut moyenner
        @param width: int, largeur de la fenêtre de temps
        @param mean: list(float), liste des moyennes mobiles de X
    """
    # Initialisation des moyennes mobiles de X
    mean = [0] * len(X)
    
    # inter: nombre de points pris en compte de chaque côté du point courant
    inter = int(width/2)
    
    for i in range(len(X)):
        if i < inter:
            mean[i] = np.mean( X[ : i + inter + 1] )
        else:
            if len(X)-i < inter:
                mean[i] = np.mean( X[ i - inter : ] )
            else:
                mean[i] = np.mean( X[i - inter : i + inter + 1] )
    return mean    
    
def detectionBurst(data, fields, width, display=False):
    """ Fonction permettant de détecter un burst de reviews dans le temps, 
        pour les données data et fields et une fenêtre de temps de largeur
        width.
        Calcule la différence entre le nombre de reviews dans le temps et la 
        moyenne mobile.
        @param data: list(str), liste des échantillons de données, chaque
                      échantillon correspondant à un avis posté.
        @param fields: list(str), liste des attributs
        @param width: int, largeur de la fenêtre de temps
        @param mean: list(float), liste des moyennes mobiles de X
        @return diff: list(float), liste des différences entre nombre de reviews
                      et moyenne mobile
    """
    # X: nombre de reviews par mois, MM: moyenne mobile de X
    X = reviewPerMonth(data, fields)
    MM = moyenneMobile(X, width)
    
    # diff: liste des différences entre nombre de reviews et moyenne mobile
    diff = np.abs( [ X[i] - MM[i] for i in range(len(X)) ] )
    
    # Affichage graphique:
    if display:
        plt.title('Nombre de reviews par mois')
        plt.plot(X)
        plt.plot(MM)
        #plt.plot(diff)
        
    return X, MM, diff


# ------------ Détection de bursts de reviews par estimation par noyau - KDE

def KDE(data, fields, h):
    """ Fonction permettant de détecter un burst de reviews dans le temps, 
        pour les données data et fields et une fenêtre de temps de largeur h,
        en se basant sur l'estimation par noyau (KDE - Kernel Density Estimation)
        @param data: list(str), liste des échantillons de données, chaque
                      échantillon correspondant à un avis posté.
        @param fields: list(str), liste des attributs
        @param h: int, largeur de la fenêtre de temps
        @return diff: list(float), liste des différences entre nombre de reviews
                      et moyenne mobile
    """
    # X: nombre de reviews par mois, 
    # X_ext: extension de X, remis sous format histogramme
    X = reviewPerMonth(data, fields)
    X_ext = []
    
    for i in range(len(X)):
        X_ext += [i]*X[i]
    
    # Lissage par estimation par noyau (KDE - Kernel Density Estimation)
    X_ext = np.array(X_ext).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(X_ext)
    
    plt.title('Lissage par estimation par noyau - KDE')
    X_plot = np.linspace(0, len(X))
    Y_plot = kde.score_samples(X_plot.reshape(-1,1))
    plt.plot(X_plot, Y_plot)
    
    return Y_plot


############ DETECTION DE REVIEWS SPAMS PAR DEVIATION DE NOTATION ############

# Les reviews spams impliquent généralement une projection incorrecte de la
# qualité d'un produit via une note qui dévie beaucoup de l'opinion générale

def sortReviewsByTime(data, fields):
    """ Trie les reviews de l'échantillon de données data par ordre 
        chronologique.
        @param data: list(str), liste des échantillons de données, chaque
                     échantillon correspondant à un avis posté
        @param fields: list(str), liste des attributs
        @return : list(str), data avec reviews triés chronologiquement
        @return indices: list(int), liste des indices des données triés 
                         chrolonogiquement dans l'échantillon data initial
    """
    index_time = fields.index('reviewTime')
    
    # Liste des dates et des indices de ces dates
    dates = [0] * len(data)
    dates_indices = []
    
    # Temps sous la forme 'mois jour, annee'
    for i in range(len(data)):
        t = data[i][index_time]
        mois, jour, annee = t.split()
        dates[i] = (int(annee), int(mois), int(jour.strip(',')))
    
    # Tri des dates par ordre chronologique et récupération des indices des
    # dates triées
    for date, indice in sorted((e, i) for i, e in enumerate(dates)):
        dates_indices.append(indice)
    
    return [ data[i] for i in dates_indices ], dates_indices


def deviationNotes(data, fields, width, bins, seuil):
    """ Calcule la déviation des notes dans le temps et crée l'histogramme
        correspondant aux déviations entre notes données par les utilisateurs
        note moyenne (moyenne mobile).
        @param data: list(str), liste des échantillons de données, chaque
                     échantillon correspondant à un avis posté
        @param fields: list(str), liste des attributs
        @param width: int, largeur de la fenêtre pour la moyenne mobile
        @param bins: int, nombre d'intervalles dans [0,5] que l'on veut 
                     considérer pour tracer l'histogramme
        @return : float array, effectif des différences par intervalle
    """
    # Tri des reviews de data par ordre chronologique
    data_sorted, indices_sorted = sortReviewsByTime(data, fields)
    
    # Notes des reviews triés chronologiquement, calcul des notes moyennes
    notes = np.array( [ r[0] for r in data_sorted ] )
    notes_moyennes = np.array( moyenneMobile(notes, width) )
    
    # Calcul de la déviation entre notes données et notes moyennes (moyenne mobile)
    deviation_notes = np.abs(notes - notes_moyennes)
    
    # Affichage par histogramme, récupération des effectifs par intervalle
    plt.title('Histogramme de la déviation des notes, pour bins = %d' %bins)
    hist = plt.hist(deviation_notes, np.linspace(0, 5, bins+1), color='thistle')
    plt.xlabel('Note')
    plt.ylabel('Nombre de revues')
    
    # retourner les dates des reviews dont la déviation est de plus de 3,25
    indices_suspect = np.where(deviation_notes >= seuil)[0]

    return [indices_sorted[i] for i in indices_suspect]



def showRDperMonth(data, fields, width, bins, seuil):
    """ Renvoie la liste du nombre de reviews posté par mois dans data.
        @param data: list(str), liste des échantillons de données, chaque
                      échantillon correspondant à un avis posté.
        @param fields: list(str), liste des attributs
        @return X: list(int), liste du nombre de reviews par mois
    """
    # ----------------- TRACAGE DU NOMBRE DE REVIEWS PAR MOIS
    
    index_time = fields.index('reviewTime')
    
    # Temps sous la forme 'mois jour, annee'
    times = [d[index_time] for d in data]
    
    # Normalisation sous la forme (annee, mois, jour)
    times_normalized = []
    for t in times:
        mois, jour, annee = t.split()
        times_normalized.append((int(annee), int(mois), int(jour.strip(','))))
    
    # Tri des temps du plus ancien au plus récent
    times_normalized = sorted(times_normalized)
    
    # Plot le nombre de review par mois
    liste_annees = sorted( np.unique( [t[0] for t in times_normalized] ) )
    liste_mois = sorted( np.unique( [t[1] for t in times_normalized] ) )
    liste_jours = sorted( np.unique( [t[2] for t in times_normalized] ) )
    
    # X: liste du nombre de reviews postés par mois
    X = [0] * (len(liste_annees) * len(liste_mois))
    
    for annee, mois, jour in times_normalized:
        X[liste_annees.index(annee)*len(liste_mois) + liste_mois.index(mois)] += 1
    
    
    # ----------------- AFFICHAGE DES DATES OU L'ON SUSPECTE UN SPAM
    
    index_dev = deviationNotes(data, fields, width, bins, seuil)
    data_dev = [data[i] for i in index_dev]
    
    # X: liste des dates où l'on suspecte un review spam
    Y = [0] * (len(liste_annees) * len(liste_mois))
    
    # Temps sous la forme 'mois jour, annee'
    times = [d[index_time] for d in data_dev]
    
    # Normalisation sous la forme (annee, mois, jour)
    times_normalized = []
    for t in times:
        mois, jour, annee = t.split()
        times_normalized.append((int(annee), int(mois), int(jour.strip(','))))
    
    # Tri des temps du plus ancien au plus récent
    times_normalized = sorted(times_normalized)
    
    for annee, mois, jour in times_normalized:
        Y[liste_annees.index(annee)*len(liste_mois) + liste_mois.index(mois)] += 1
    
    # ------- Moyenne mobile
    
    mm = moyenneMobile(X, 5)
    
    # ------- AFFICHAGE
    plt.figure()
    # Calculer l'histogramme
    rpm = reviewPerMonth(data, fields)
    hst = []
    for n in range(len(rpm)):
        if rpm[n] != 0:
            hst += rpm[n] * [n]
    
    # Tracer l'histogramme
    plt.figure()
    plt.title('Rating deviation')
    # plt.hist(hst, color = 'midnightblue')
    plt.xlabel('Mois')
    
    
    plt.plot(X, color='lightgreen', label='Nombre de revues de déviation >= s')
    plt.plot(mm, color='darkgreen', label='Moyenne mobile')
    # for e in Y:
    #     if e != 0:
    #         plt.scatter(Y.index(e), 0, s=20, zorder=2, color = 'red')
    plt.legend()
    plt.show()
    
    return X, Y, mm


def doublonsNonSpams(data, fields, nomFichier = 'doublons_non_spams.txt'):
    """
        List * List * str -> List
        data : liste de review (liste)
        fields : liste des attributs de data
        nomFichier: path ou est stocké la liste sans doublon
        
        renvoie data sans les doublons non-spams (auteur, texte, produit) et la stock dans nomFichier
    
    """
    # Compteur de doublons
    cpt = 0
    
    if fields.index('reviewerName'):
        iAuteur=fields.index('reviewerName')
        if fields.index('reviewText'):
            iText = fields.index('reviewText')
            if fields.index('asin'):
                iProd = fields.index('asin')
            else:
                return 0
        else:
            return 0
    else:
        return 0
    
    
    hashs=set()
    file= open(nomFichier, 'w', encoding='utf-8')
    data2=[]   
    
    for i in data:
        if i[iText]:
            try:
                h = hash(i[iAuteur]+i[iText]+i[iProd])
            except:
                h = hash(i[iText]+i[iProd])
        else:
            continue
        
        if h not in hashs:
            hashs.add(h)
            json.dump(i, file)
            data2.append(i)
        else:
            cpt += 1
    
    # On affiche le nombre de doublons détectés
    print('\n', str(cpt) , 'doublons non-spams (idReviewer - review - produit) détectés')
    file.close()
    return data2


def doublonsSpams(data, fields):
    """
        List * List * str -> List
        data : liste de review (liste)
        fields : liste des attributs de data
        nomFichier: path ou est stocké la liste sans doublon
        
        renvoie data sans les doublons (auteur, texte) et la stock dans nomFichier
    
    """
    # Compteur de doublons
    cpt = 0
    
    if fields.index('reviewText'):
        iText = fields.index('reviewText')
        if fields.index("asin"):
            iId = fields.index("asin")
        else:
            return 0
    else:
        return 0
    
    
    hashs=dict()
    res=set()  
    
    for i in range(len(data)):
        if data[i][iText]:
            h = hash(data[i][iText])
        else:
            continue
        
        if h not in hashs:
            hashs[h] = i
        else:
            if len(data[i][6]) > 20:
                res.add(hashs[h])
                res.add(i)
                cpt += 1
    
    # On affiche le nombre de doublons détectés
    print('\n', str(cpt) , 'doublons spams (<> idReviewer, <> idProduit, review) détectés')
           
    return res


def detectFromBurstRD(data, fields, width, bins, seuil, window=4, height=50, distance=10, display=False):
    """ Renvoie la liste des indices des datas de reviews suspects:
        * reviews dont la note dévie beaucoup de la moyenne mobile
        * et qui en plus se trouve près d'un pic (burst de reviews)
        @param width: int, largeur de la fenêtre pour la moyenne mobile
        @param bins: int, nombre d'intervalles dans [0,5] que l'on veut 
                     considérer pour tracer l'histogramme
        @param seuil: float, valeur à partir de laquelle une déviation est suspecte
        @param window: int, fenetre de temps que l'on considère suspect autour 
                       d'un peak du nombre de reviews
        @param height: int, seuil au dessus duquel on prend nos pics
        @param distance: int, distance min entre 2 pics
        
        @return suspams_ids: list(int), liste des indices des reviews suspectées
                             spam (dans fenêtre de burst + déviation note)
    """
    # Première liste de reviews suspectes: reviews dans un burst temporel
    X, Y, mm = showRDperMonth(data, fields, width, bins, seuil)
    X = np.array(X)
    Y = np.array(Y)
    
    # Trouver la liste les maxima locaux se trouvant au dessus d'un certain seuil height,
    # et à une certaine distance d'autres peaks
    peaks, _ = find_peaks(X, height = height, distance = distance)
    
    # Dans peaks on a maintenant la liste indices des pics (maxima locaux) du 
    # nombre de reviews par mois. On considère les reviews dont les indices se
    # trouvent dans une fenêtre de temps window autour de ces pics
    
    # Liste des indices de pics et leur voisinage direct
    window_ids = []
    w = window // 2
    
    for p in peaks:
        window_ids += [p + i for i in range(-w , w + 1)]
    
    Y = np.where(Y != 0)[0]
    
    # windows_ids: liste des indices de reviews dans une fenêtre temporelle d'un pic
    # Y: liste des indices de reviews dont la note est suspecte (déviation forte)
    # On veut les indices communs des deux listes
    suspams = [i for i in Y if i in window_ids]
    
    # Affichage
    if display:
        # Calculer l'histogramme
        rpm = reviewPerMonth(data, fields)
        hst = []
        for n in range(len(rpm)):
            if rpm[n] != 0:
                hst += rpm[n] * [n]
        
        # ----------------- Premier graphe: pics de revues -------------------
        
        # Tracer l'histogramme
        plt.figure()
        plt.title('Bursts de revues spams')
        plt.hist(hst, color = 'midnightblue')
        plt.xlabel('Mois')
        plt.ylabel('Nombre de revues')
        
        # Affichage des pics
        plt.plot(X, color='palegreen', label='Nombre de revues déviantes')
        plt.plot(peaks, X[peaks], "X", color='orange', label='Burst spams')
        plt.legend()
        
        
        # ---------------- Deuxième graphe: moyenne mobile -------------------
        
        plt.figure()
        plt.title('Moyenne mobile et détection de revues spams')
        
        # Calculer l'histogramme
        rpm = reviewPerMonth(data, fields)
        hst = []
        for n in range(len(rpm)):
            if rpm[n] != 0:
                hst += rpm[n] * [n]
        
        # Tracer l'histogramme
        #plt.title('Nombre de revues par mois pour le produit')
        plt.hist(hst, color = 'midnightblue')
        plt.xlabel('Mois')
        plt.ylabel('Nombre de revues')
        
        # Affichage des pics
        plt.plot(mm, color='lightsteelblue', label='Moyenne mobile sur les revues déviantes')
        
        # Affichage des reviews suspectées comme spam
        for r in suspams[:-1]:
            plt.scatter(r, 0, s=10, zorder=2, color='red', clip_on=False)
        plt.scatter(suspams[-1], 0, s=10, zorder=2, color='red', clip_on=False, label='Revues suspectes')
        plt.legend()
        plt.show()
    
    # Temps sous la forme 'mois jour, annee'
    index_time = fields.index('reviewTime')
    times = [d[index_time] for d in data]
    
    # Normalisation sous la forme (annee, mois, jour)
    tn = []
    for t in times:
        mois, jour, annee = t.split()
        tn.append((int(annee), int(mois), int(jour.strip(','))))    
    
    liste_annees = sorted( np.unique( [t[0] for t in tn] ) )
    liste_mois = sorted( np.unique( [t[1] for t in tn] ) )
    liste_jours = sorted( np.unique( [t[2] for t in tn] ) )
    
    # suspams_ids: Liste des indices des reviews suspectes dans data
    suspams_ids = [i for i in range(len(data)) if liste_annees.index(tn[i][0])*len(liste_mois) + liste_mois.index(tn[i][1]) in suspams]
    
    return set(suspams_ids)


######## ALGORITHME TYPE PAGERANK POUR LA DETECTION DE REVIEWS SPAMS #########

# La classe ci-dessous implémente un algorithme de type PageRank dont le but 
# est d'attribuer un score de spammicité aux produits, utilisateurs et revues.
# Comprises entre -1 et 1, ces scores de spammicité indiquent dans le cas où le
# score est très proche de -1:
#   * un produit attaqué par des utilisateurs malhonnêtes
#   * des utilisateurs malhonnêtes
#   * des revues spams

class ReviewGraph:
    """ Classe pour la propagation des scores utilisateur - score - produit
        par un algorithme de type PageRank.
    """
    def __init__(self, data, fields, width=6, bins=20, seuil=3.25, window=1, height=50, distance=10):
        """ Constructeur de la classe ReviewGraph.
        """
        self.data = data
        self.fields = fields
        
        # ID des reviews qui sont à fois dans un burst et dont la note dévie beaucoup de la moyenne
        self.suspams_ids = detectFromBurstRD(data, fields, width=width, bins=bins, seuil=seuil, window=window, height=height, distance=distance, display=False)
        
        # ID des reviews doublons potentiellement spams
        self.doublons_ids = doublonsSpams(data, fields)
        
        # Initialisation des scores utilisateur, produits et reviews
        
        # T: Reviewer's trustiness (score utilisateur): initialisation à +1
        self.index_utils = fields.index('reviewerID')
        unique_utils = np.unique( np.array( [r[self.index_utils] for r in data] ) )
        self.score_utils = { id_util : 1 for id_util in unique_utils }
        
        # R: Product reliability (score produit): initialisation à +1
        self.index_prods = fields.index('asin')
        unique_prods = np.unique( np.array( [r[self.index_prods] for r in data] ) )
        self.score_prods = { id_prod : 1 for id_prod in unique_prods }
        
        # H: Review honesty (score review): initialisation selon le score d'entente
        self.score_revs = { id_rev : 0 for id_rev in range( len( data ) ) }
        
        # Liste des auteurs de revues en doublons ou dans un burst + déviation
        for ids in self.suspams_ids.union(self.doublons_ids):
            self.score_utils[data[ids][self.index_utils]] = -0.02 # biais
            
        # Création du dictionnaire des revues par mois: idRevsDate
        self.index_time = fields.index('reviewTime')
        self.idRevsDate = {}
        
        for i in range(len(data)):
            mois, jour, annee = data[i][self.index_time].split()
            if (int(annee), int(mois)) not in self.idRevsDate:
                self.idRevsDate[ (int(annee), int(mois)) ] = []
            self.idRevsDate[ (int(annee), int(mois)) ].append(i)
        
        # Clés de idRevsDate dans l'ordre chronologique
        self.idRevsDate_keys = sorted( self.idRevsDate.keys() )
        
        # Indice de la note dans une revue
        self.index_ratings = fields.index('overall')
        
        # Pour nous faciliter la tâche, on redéfinit la base de données en 
        # indexant par utilisateur ou par produit
        self.db_utils = { id_util : [] for id_util in self.score_utils }
        self.db_prods = { id_prod : [] for id_prod in self.score_prods }
        
        for i in range(len(data)):
            self.db_utils[ data[i][self.index_utils] ].append( (i, data[i]) )
            self.db_prods[ data[i][self.index_prods] ].append( (i, data[i]) )
        
        # Calcul des voisins temporels de chaque revue
        self.neighs = { id_rev : self.getNeighRevs(id_rev, window = window) for id_rev in self.score_revs }
        
        # Initialisation du score d'entente
        self.score_entente = self.initScoreRevs()
        
        # Nombre d'itérations
        self.niter = 0
        
    def getNeighRevs(self, index_rev, window = 4):
        """ Pour trouver les revues publiées dans une fenêtre de temps window
            autour de la revue d'indice i dans data (fenêtre en mois: si elle
            vaut 6 par exemple, on prend les revues publiées entre les 3 mois 
            précédant la revue actuelle et les 3 mois suivant sa publication).
            @param index_rev: int, indice du review dans data
            @param window: int, fenêtre de temps (en mois) considérée
            @return neighRevs: list(int), liste des indices des voisins de 
                               index_rev dans la fenêtre de temps window.
        """
        w = window // 2
        
        # On retrouve l'indice de la revue dans self.idRevsDate
        mois, jour, annee = self.data[index_rev][self.index_time].split()
        i = self.idRevsDate_keys.index( (int(annee), int(mois)) )
        
        # Définition des indices bornant la fenêtre de temps
        imin = i - w if i - w > 0 else 0
        imax = i + w + 1 if i + w + 1 < len( self.idRevsDate_keys ) else len( self.idRevsDate_keys )
        
        # neighRevs: liste des revues publiées dans la fenêtre de temps
        neighRevs = []
        for i in self.idRevsDate_keys[ imin : imax ]:
            neighRevs += self.idRevsDate[i]
        
        return neighRevs
    
    def T(self, id_util):
        """ Reviewer's trustiness (confiance utilisateur).
        """
        # Calcul de la somme des scores honnêteté des revues de l'utilisateur
        h = np.sum( [ self.score_revs[id_rev] for id_rev, rev in self.db_utils[id_util] ] )
        return ( 2 / ( 1 + np.exp(-h) ) ) - 1
        
    def A(self, id_rev, delta = 1):
        """ Agreement score (score d'entente).
            @param delta: float, borne sur la différence entre la note actuelle 
                          et les notes des voisins.
        """
        # On récupère les indices des voisins (dans le sens temporel) de id_rev
        neighs_temp = self.neighs[id_rev]
        
        # Produit correspondant à la revue courante
        id_prod = self.data[id_rev][self.index_prods]
        
        # Parmi les revues voisines, on ne garde que celles qui traitent du même produit
        neighs = []
        
        for id_neigh in neighs_temp:
            if self.data[id_neigh][self.index_prods] == id_prod:
                neighs.append(id_neigh)
                
        # Liste des voisins de notes similaires ou différentes à id_rev
        index_rating = self.fields.index('overall')
        rating = self.data[id_rev][index_rating]
        
        # Compteurs score de confiance des voisins similaires - différents
        cpt_sim = 0
        cpt_dif = 0
        
        for i in neighs:
            if np.abs( rating - self.data[i][index_rating] ) <= delta:
                cpt_sim += self.score_utils[self.data[i][self.index_utils]]
            else:
                cpt_dif += self.score_utils[self.data[i][self.index_utils]]
        
        # Calcul et normalisation du score d'entente
        a = cpt_sim - cpt_dif
        a = ( 2 / ( 1 + np.exp(-a) ) ) - 1
        
        return a
    
    def H(self, id_rev, delta = 1):
        """ Review honesty (honnêteté revue).
            @param delta: float, borne sur la différence entre la note actuelle 
                          et les notes des voisins.
        """
        return np.abs( self.score_prods[ self.data[id_rev][self.index_prods] ] ) * self.score_entente[id_rev]
                      
    def R(self, id_prod, median = 3):
        """ Product reliability (fiabilité produit).
        """        
        theta = 0
        for i, d in self.db_prods[id_prod]:
            if self.score_utils[ d[self.index_utils] ] > 0:
                theta += self.score_utils[ d[self.index_utils] ] * ( d[self.index_ratings] - median )
        
        return ( 2 / ( 1 + np.exp(-theta) ) ) - 1
    
    def initScoreRevs(self):
        """ Initialisation des scores revues.
            Valeurs:
                * -1 si à la fois doublon, dans un burst et note déviante.
                * -0.75 si doublon ou note déviante (variante)
                * sinon: selon le score d'entente
        """
        score_entente = {}
        suspams = self.suspams_ids.intersection(self.doublons_ids)
        print('%d doublons spams dans bursts détectés' % len(suspams))
        
        for id_rev in range( len( self.data ) ):
            if id_rev in suspams:
                score_entente[id_rev] = -1
            else:
                score_entente[id_rev] = self.A(id_rev, delta = 1)
        
        return score_entente
    
    def computeScores(self, delta = 1, median = 3, niter = 50):
        """ Algorithme de type PageRank pour propager les scores.
            @param window: int, fenêtre de temps prise pour prendre le 
                           voisinage d'une revue
            @param delta: float, borne sur la différence entre la note actuelle 
                          et les notes des voisins.
            @param median: float, valeur médiane des notes
            @param niter: int, nombre d'itérations
        """
        self.niter += niter
        
        for iter in range(niter):
            
            # Propagation scores reviews
            for id_rev in self.score_revs:
                self.score_revs[id_rev] = self.H(id_rev, delta = delta)
            
            # Propagation des scores utilisateurs
            for id_util in self.score_utils:
                self.score_utils[id_util] = self.T(id_util)
            
            # Propagation des scores produits
            for id_prod in self.score_prods:
                self.score_prods[id_prod] = self.R(id_prod, median = median)
        
        # On remet les scores des reviews sur 100
        self.score_revs_norms = { id_rev : np.round( (self.score_revs[id_rev] + 1) * 50 ) for id_rev in self.score_revs }

        return self.score_revs_norms 
    
    def display_scores(self):
        """ Affichage des histogrammes des scores.
        """
        plt.figure()
        plt.title('Scores utilisateurs, pour niter = %d' % self.niter)
        plt.xlim(-1,1.1)
        plt.hist( list( self.score_utils.values() ), color='darkgreen' )
        plt.xlabel('Score de confiance')
        plt.ylabel('Nombre de revues')
        plt.show()
        
        
        plt.figure()
        plt.title('Scores produits, pour niter = %d' % self.niter)
        plt.xlim(-1,1.1)
        plt.hist( list( self.score_prods.values() ), color='midnightblue' )
        plt.xlabel('Score de fiabilité')
        plt.ylabel('Nombre de revues')
        plt.show()
        
        plt.figure()
        plt.title('Scores reviews, pour niter = %d' % self.niter)
        plt.xlim(-1,1.1)
        plt.hist( list( self.score_revs.values() ), color='indigo' )
        plt.xlabel('Score d\'honnêté')
        plt.ylabel('Nombre de revues')
        plt.show()
    
    def detect_susprods(self, seuil=-0.75):
        """ Retourne la liste des produits suspects, ie dont le score est 
            inférieur au seuil.
        """
        return [ k for k , v in self.score_prods.items() if v < seuil ]
    
    def detect_susrevs(self, seuil=-0.75):
        """ Retourne la liste des produits suspects, ie dont le score est 
            inférieur au seuil.
        """
        return [ k for k , v in self.score_revs.items() if v < seuil ]
    
    def get_k_worst(self, k):
        """ Retourne la liste des k produits de plus mauvais scores.
        """
        # Tri dans l'ordre croissant des scores
        scores_sorted = { k : v for k, v in sorted(self.score_prods.items(), key=lambda item: item[1]) }
        return list( scores_sorted.keys() )[:k]
        
    def get_k_best(self, k):
        """ Retourne la liste des k produits de meilleurs scores.
        """
        # Tri dans l'ordre décroissant des scores
        scores_sorted = { k : v for k, v in sorted(self.score_prods.items(), key=lambda item: item[1], reverse=True) }
        return list( scores_sorted.keys() )[:k]
    
    def prod_timeline(self, id_prod):
        """ Trace l'évolution de la moyenne des notes d'un produit dans le 
            temps. Prend en compte toutes les notes depuis le début.
        """
        # On isole les données concernant le produit
        data_prod = [ d for d in self.data if d[self.index_prods] == id_prod ]
        
        # On trie dans l'ordre chronologique
        data_prod = sortReviewsByTime(data_prod, self.fields)[0]
        
        index_time = self.fields.index('reviewTime')
        
        # Temps sous la forme 'mois jour, annee'
        times = [d[index_time] for d in data_prod]
        
        # Normalisation sous la forme (annee, mois, jour)
        times_normalized = []
        for t in times:
            mois, jour, annee = t.split()
            times_normalized.append((int(annee), int(mois), int(jour.strip(','))))
        
        # Plot le nombre de review par mois
        liste_annees = [ t[0] for t in times_normalized ]

        liste_annees = [ i for i in range(min(liste_annees), max(liste_annees) + 1) ]
        liste_mois = [ i for i in range(1,13) ]
        
        # X: moyenne des notes par mois
        X = [[]] * (len(liste_annees) * len(liste_mois))
        
        for i in range(len(times_normalized)):
            annee, mois, jour = times_normalized[i]
            index = liste_annees.index(annee)*len(liste_mois) + liste_mois.index(mois)
            X[index] = X[index] + [data_prod[i][0]]
        
        
        # Pour chaque mois, on calcule la moyenne depuis le début jusqu'à maintenant
        X_mean = [ np.mean([ e for x in X[: i + 1] for e in x ]) for i in range(len(X))]
        X_mean = [ x if not np.isnan(x) else 0 for x in X_mean ]
        
        # Histogramme du nombre de revues par mois
        X_hist = [ [i] * len(X[i]) for i in range(len(X)) ]
        X_hist = [ e for x in X_hist for e in x ]
        
         # ------- AFFICHAGE
        fig = plt.figure(figsize=(20,5))
        plt.title('Product timeline idprod = %s' %id_prod, y = 1.2)
        plt.axis('off')
        
        ax1 = fig.add_subplot(121)
        ax1.title.set_text('Evolution de la note moyenne dans le temps')
        ax1.set_ylim([0,5])
        ax1 = plt.plot(X_mean, color='midnightblue')
        
        ax2 = fig.add_subplot(122)
        ax2.title.set_text('Nombre de revues par mois')
        ax2 = plt.hist(X_hist, color='lightsteelblue')
        
    def getScoresRevs(self):
        return self.score_revs
    
    def getScoresUtils(self):
        return self.score_utils
    
    def getScoresProds(self):
        return self.score_prods
    
    def getScoresRevsNorms(self):
        return self.score_revs_norms



############ PARTIE TAL: IMPORTATION DES LIBRAIRIES ET MODULES ###############

import string
import unicodedata

from nltk.corpus import stopwords as stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import SGDClassifier


################ DETECTION D'ATTAQUES PAR DES TECHNIQUES TAL #################

class TextProcessor:
    """ Classe pour pré-traiter les données textuelles (revues dans notre cas)
        avant d'appliquer des méthodes discriminantes ou de clustering.
    """
    def __init__(self, corpus):
        """ Constructeur de la classe TextProcessor.
            @param corpus: str array, corpus de textes
        """
        self.corpus= corpus
        self.vectorizer = None
        self.sim_matrix = None
        self.sim_threshold = None
        self.sum_sim_threshold = None
        self.coef_ = None
        self.main_features = None
    
    def process(self, language='english', lower=True, remove_punc=True, remove_digits=True, normalize=True, remove_stopwords=True, stemming=True):
        """ Pré-traitement paramétré du corpus de textes passé en argument.
            @param language: str, langage utilisé (utile pour les stopwords)
            @param lower: bool, True si l'on met tout le texte en minuscule
            @param remove_punc: bool, True si l'on supprime la ponctuation
            @param remove_digits: bool, True si l'on supprime les chiffres
            @param normalize: bool, True si l'on normalise le texte
            @param remove_stopwords: bool, True si l'on supprime les stopwords
            @param stemming: bool, True si l'on applique le stemming
        """
        # Liste des ponctuations à supprimer
        punc = string.punctuation + '\n\r\t'
        
        # Liste des stopwords à supprimer (en anglais)
        english_stopwords = set(stopwords.words(language))
        
        # Stemmer anglais
        stemmer = SnowballStemmer(language, ignore_stopwords=True)
        
        # Copie profonde du corpus de textes
        self.processed_corpus = copy.deepcopy(self.corpus)
        
        for i in range(len(self.corpus)):
            text = self.processed_corpus[i]
            if lower: text = text.lower()
            if remove_punc: text = text.translate(str.maketrans(punc, ' ' * len(punc)))
            if remove_digits: text = text.translate(str.maketrans('', '', string.digits))
            if normalize: text = unicodedata.normalize(u'NFKD', text).encode('ascii', 'ignore').decode('utf8')
            if remove_stopwords: text = ' '.join([ word for word in text.split() if word not in english_stopwords ])
            if stemming: text = ' '.join([ stemmer.stem(word) for word in text.split() ])
            self.processed_corpus[i] = text
        
        return self.processed_corpus
    
    def vectorize(self, vtype='count', analyzer='word', ngram_range=(1,1)):
        """ Calcule les représentations de chaque texte du corpus pré-traité 
            sous forme de sacs de mots. L'attribut self.X renvoyé correspond à
            la matrice des représentations, et self.features aux mots retenus.
            Pour éviter de se retrouver par la suite avec des similarités trop
            faibles dues au bruit, on ne gardera pas les mots n'apparaissant 
            qu'une seule fois dans tout le corpus.
            @param vtype: str ('count' ou 'tf-idf'), spécifie le type de 
                          représentation souhaité
            @param analyzer: str ('word' ou 'char'), spécifie la granularité
            @param ngram_range: (min:int, max:int), intervalle de n-grams
        """
        if vtype == 'count':
            self.vectorizer = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=2)
        if vtype == 'tf-idf':
            self.vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=2)
        
        # Représentation des documents et dimensions (n-grams)
        self.X = self.vectorizer.fit_transform(self.processed_corpus)
        self.features = self.vectorizer.get_feature_names()
        
        return self.X, self.features 
        
    def cosine_similarity(self, X, Y, decimals = 4):
        """ Calcule la similarité cosinus entre deux vecteurs X et Y.
            @param X: float array, vecteur (d'occurrences ou tf-idf)
            @param Y: float array, vecteur (d'occurrences ou tf-idf)
            @param decimals: int, nombre de décimales gardées
            @return : float, score de similarité entre X et Y
        """
        return round( np.dot(X,Y) / (np.linalg.norm(X) * np.linalg.norm(Y)) , decimals )
    
    def similarity_matrix(self, decimals = 4):
        """ Retourne la matrice de similarité sur tous les textes du corpus
            pré-traité. La métrique utilisée est la similarité cosinus.
            On fait attention à ne pas prendre en compte la diagonale (0).
            @param decimals: int, nombre de décimales gardées
            @return sim_matrix: (float) array x array, matrice de similarité
        """
        if self.vectorizer == None:
            raise ValueError('Cannot compute similarity matrix when vectorizer does not exist. Do make sure to use TextProcessor.vectorize function beforehand.')
        
        # Représentation des documents sous forme d'array
        X = self.X.toarray()
        
        # Initialisation de la matrice de similarités
        ntxt = len(X) # Nombre de textes dans le corpus
        self.sim_matrix = np.zeros((ntxt,ntxt))
        
        for i in range(ntxt):
            for j in range(ntxt):
                if i != j:
                    self.sim_matrix[i][j] = self.cosine_similarity( X[i], X[j], decimals=decimals )
        
        # Remplacer les valeurs NaN par 0.
        np.nan_to_num(self.sim_matrix, copy=False, nan=0.0)
        
        return self.sim_matrix
    
    def similarity_threshold(self, threshold, decimals = 4):
        """ Retourne la matrice de similarité sur tous les textes du corpus
            pré-traité, à laquelle on a appliqué un seuil. Elle vaut 1 en les
            points où le score de similarité est au moins égal au seuil.
            @param threshold: float, seuil
            @param decimals: int, nombre de décimales gardées
            @return self.sim_threshold: (float) array x array, matrice de 
                                        similarité seuillée
        """
        self.sim_matrix = self.similarity_matrix(decimals=decimals)
        self.sim_threshold = copy.deepcopy(self.sim_matrix)
        
        # Seuillage
        self.sim_threshold[ self.sim_matrix < threshold ] = 0
        self.sim_threshold[ self.sim_matrix >= threshold ] = 1
        
        return self.sim_threshold
    
    def overall_threshold(self, threshold):
        """ Cette fois-ci, on travaille depuis la patrice de similarité seuillée.
            La fonction retourne 1 si la somme (seuillée) des similarités sur 
            un document est au moins égale au seuil epsilon.
            @param threshold: int, seuil
        """
        if self.sim_threshold.all() == None:
            raise ValueError('Cannot compute overall threshold when thresholded similarity matrix has yet to be computed. Please use TextProcessor.similarity_threshold function beforehand.')
        
        self.sum_sim_threshold = np.sum(self.sim_threshold, axis=1)
        self.sum_sim_threshold[ self.sum_sim_threshold < threshold ] = 0
        self.sum_sim_threshold[ self.sum_sim_threshold >= threshold ] = 1
        
        return self.sum_sim_threshold
    
    
    def feature_importance(self, coef, features, top_features=5):
        """ Pour un aperçu visuel des features les plus importants.
        """
        coef_ = coef.ravel()
         
        # Features importants pour chaque classe
        top_positive_coefficients = np.argsort(coef_)[-top_features:]
        top_negative_coefficients = np.argsort(coef_)[:top_features]
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        
        # Affichage des features les plus importants
        plt.figure(figsize=(15, 5))
        plt.title('Features discriminants ', y = 1.2 )
        colors = ['palegreen' if c < 0 else 'thistle' for c in coef_[top_coefficients]]
        plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
        features = np.array(features)
        plt.xticks(np.arange(1, 1 + 2 * top_features), features[top_coefficients], rotation=60, ha='right')
        plt.show()
    
    def discriminant_features(self, threshold=None, display=True, top_features=5):
        """ Retrouve les mots (n-grams) discriminants en appliquant un classifieur
            SVM linéaire avec une régularisation Elastic Net.
        """
        if type(threshold) != np.ndarray:
            threshold = self.sum_sim_threshold
        
            
        # Données d'apprentissage
        xtrain = self.X.toarray()
        ytrain = threshold
        
        # Phase d'apprentissage
        clf = SGDClassifier(loss='hinge', penalty='elasticnet')
        clf.fit(xtrain, ytrain)
        self.coef_ = clf.coef_[0]
        
        # Sélection des attributs dscriminants (mots, n-grams)
        self.main_features = [ self.features[i] for i in range(len(self.coef_)) if self.coef_[i] > 0 ]
        
        if display:
            self.feature_importance(self.coef_, self.features, top_features)
        
        return self.main_features
    
def prod_revs_per_month(data, fields, id_prod):
    """ Renvoie le dictionnaire des revues concernant le produit id_prod indexé
        par mois.
        @param id_prod: str, identifiant produit
        @return prod_revs: dict(str, list(str)), dictionnaire des revues par mois
    """
    # Indices des identifiants produits et des dates de publication dans fields
    index_prods = fields.index('asin')
    index_time = fields.index('reviewTime')
    index_revs = fields.index('reviewText')
    
    # Liste des revues concernant le produit id_prod, triée chronologiquement
    data_prod = [ d for d in data if d[index_prods] == id_prod ]
    data_prod = sortReviewsByTime(data_prod, fields)[0]
    
     # Temps sous la forme 'mois jour, annee'
    times = [d[index_time] for d in data_prod]
            
    # Normalisation sous la forme (annee, mois, jour)
    times_normalized = []
    for t in times:
        mois, jour, annee = t.split()
        times_normalized.append((int(annee), int(mois)))
    
    # On prend aussi en compte les mois avec 0 revues
    liste_annees = [ time[0] for time in times_normalized ]
    liste_annees = [ i for i in range(min(liste_annees), max(liste_annees) + 1) ]
    liste_mois = [ i for i in range(1,13) ]
        
    # all_times: moyenne des notes par mois
    all_times = [ (annee, mois) for annee in liste_annees for mois in liste_mois ]
    
    # Initialisation du dictionnaire des revues par mois
    prod_revs = { time : [] for time in all_times }
    
    for rev in data_prod:
        mois, jour, annee = rev[index_time].split()
        prod_revs[ (int(annee), int(mois)) ].append( rev[index_revs] )
    
    return prod_revs


def create_corpus(data, fields, id_prod, month_min=None, month_max=None):
    """ Crée le corpus de revues du produit id_prod entre les mois month_min 
        et month_max.
    """
    corpus = []
    prod_revs = prod_revs_per_month(data, fields, id_prod)
    
    if month_min == None:
        month_min = 0
    if month_max == None:
        month_max = len(prod_revs)
   
    keys = list( prod_revs.keys() )
    for i in range(month_min, month_max):
        corpus += prod_revs[keys[i]]
        
    print('Création du corpus pour le produit {} entre les mois {} et {}: {} revues\n' . format(id_prod, month_min, month_max, len(corpus)))
    return corpus


def etude_1(data, fields, id_prod, month_min=None, month_max=None, vtype='tf-idf', analyzer='word', ngram_range=(1,1), display_features=True, top_features=5):
    """ Première méthode proposée.
    """
    corpus = create_corpus(data, fields, id_prod, month_min, month_max)
    
    tp = TextProcessor(corpus)
    tp.process(lower=True, remove_punc=True, remove_digits=True, normalize=True, remove_stopwords=True, stemming=True)
    X, features = tp.vectorize(vtype=vtype, analyzer=analyzer, ngram_range=ngram_range)
    
    # Calcul et affichage de la matrice de similarité cosinus
    sim_matrix = tp.similarity_matrix()
    
    # On demande à fixer un seuil sur la mesure de similarité
    plt.figure()
    plt.title('Histogramme sur la similarité cosinus')
    plt.hist( np.max(sim_matrix, axis=1), color='lightsteelblue' )
    plt.show()
    
    threshold_1 = float( input('Seuil sur la similarité: ') )
    sim_threshold = tp.similarity_threshold( threshold = threshold_1 )
    
    # On demande à fixer un seuil sur la somme des similarités binaires
    plt.figure()
    plt.title('Histogramme sur la somme des similarités seuillées, seuil = %f' % threshold_1)
    plt.hist( np.sum(sim_threshold, axis=1), color='lightsteelblue' )
    plt.show()
    
    threshold_2 = float( input('Seuil sur la somme des similarités seuillées: ') )
    sum_sim_threshold = tp.overall_threshold( threshold = threshold_2 )
    
    # Affichage des matrices de similarité
    fig = plt.figure(figsize=(20,5))
    plt.title('Matrices de similarité, id_prod = %s' % id_prod, y = 1.2)
    plt.axis('off')
        
    ax1 = fig.add_subplot(121)
    ax1.title.set_text('sim_matrix')
    ax1 = plt.imshow(sim_matrix, cmap='Greens', vmin=0, vmax=1)
        
    ax2 = fig.add_subplot(122)
    ax2.title.set_text('sim_threshold, seuil à %f' % threshold_1)
    ax2 = plt.imshow(sim_threshold, cmap='Greens', vmin=0, vmax=1)
    
    # Affichage des features discriminants
    main_features = tp.discriminant_features(display=True, top_features=top_features)
    
    #for i in range(len(sum_sim_threshold)):
    #    if sum_sim_threshold[i]==1:
    #        print('\n\n', corpus[i])
    
    return tp

def etude_1_bis(data, fields, id_prod, start_burst=None, end_burst=None, start_nb=None, end_nb=None, vtype='tf-idf', analyzer='word', ngram_range=(1,1), display_features=True, top_features=5):
    """ Première méthode proposée.
    """
    corpus_burst = create_corpus(data, fields, id_prod, start_burst, end_burst)
    tp_burst = TextProcessor(corpus_burst)
    tp_burst.process(lower=True, remove_punc=True, remove_digits=True, normalize=True, remove_stopwords=True, stemming=True)
    X_burst, features_burst = tp_burst.vectorize(vtype=vtype, analyzer=analyzer, ngram_range=ngram_range)
    
    corpus_nb = create_corpus(data, fields, id_prod, start_nb, end_nb)
    tp_nb = TextProcessor(corpus_nb)
    tp_nb.process(lower=True, remove_punc=True, remove_digits=True, normalize=True, remove_stopwords=True, stemming=True)
    X_nb, features_nb = tp_nb.vectorize(vtype=vtype, analyzer=analyzer, ngram_range=ngram_range)
    
    # Calcul et affichage de la matrice de similarité cosinus
    sim_matrix_burst = tp_burst.similarity_matrix()
    sim_matrix_nb = tp_nb.similarity_matrix()
    
    # Moyenne des similarités non nulles hors-burst
    nb_mean = np.nanmean(np.where(sim_matrix_nb!=0, sim_matrix_nb, np.nan))
    sim_matrix = np.where(sim_matrix_burst > nb_mean, sim_matrix_burst - nb_mean, 0)
    
    # On demande à fixer un seuil sur la mesure de similarité
    plt.figure()
    plt.title('Matrice de similarité en période de burst')
    plt.imshow( sim_matrix_burst, cmap='Greens', vmin=0, vmax=1 )
    plt.show()
    
    plt.figure()
    plt.title('Matrice d\'écart de similarité à la moyenne en période de burst')
    plt.imshow( sim_matrix, cmap='Greens', vmin=0, vmax=1 )
    plt.show()
    
    plt.figure()
    plt.title('Histogramme des écarts de similarité à la moyenne')
    plt.hist( np.sum(sim_matrix, axis=1), color='lightsteelblue' )
    plt.show()
    
    threshold = float( input('Seuil sur les écarts de similarité: ') ) 
    
    sim_matrix = np.where( sim_matrix > float(threshold), 1, 0 )

    plt.figure()
    plt.title('Matrice d\'écart de similarité à la moyenne binarisée')
    plt.imshow( sim_matrix, cmap='Greens', vmin=0, vmax=1 )
    plt.show()
    
    # Labellisation: 1 si la somme des similarités sur une ligne > 0, 0 sinon
    threshold_matrix = np.where( sim_matrix.sum( axis = 1 ) > 0, 1, 0 )
    
    plt.figure()
    plt.title('Label par review')
    plt.imshow( threshold_matrix.reshape(-1,1), cmap='Greens', vmin=0, vmax=1 )
    plt.show()
    
    # Affichage des features discriminants
    main_features = tp_burst.discriminant_features(threshold=threshold_matrix, display=True, top_features=top_features)
    
    #for i in range(len(sum_sim_threshold)):
    #    if sum_sim_threshold[i]==1:
    #        print('\n\n', corpus[i])
    
    return tp_burst

def etude_2(data, fields, id_prod, month_min, month_max, vtype='tf-idf', analyzer='word', ngram_range=(1,1)):
    """ Deuxième méthode proposée.
    """
    corpus = create_corpus(data, fields, id_prod)
    
    tp = TextProcessor(corpus)
    tp.process(lower=True, remove_punc=True, remove_digits=True, normalize=True, remove_stopwords=True, stemming=True)
    X, features = tp.vectorize(vtype=vtype, analyzer=analyzer, ngram_range=ngram_range)

    # On labellise les données: 1 pour les revues se trouvant dans les bursts, 0 sinon
    """ Revoir la labellisation burst: ne prendre que les revues détectées comme burst + déviation 
        sur le produit
    """
    prod_revs = prod_revs_per_month(data, fields, id_prod)
    burst_reviews = []
    
    keys = list( prod_revs.keys() )
    for i in range(month_min, month_max):
        burst_reviews += prod_revs[keys[i]]
    
    print('%d revues burst pour un corpus de taille %d.' % (len(burst_reviews), len(corpus)))
    
    Y = np.zeros(X.shape[0]) #labels
    
    for i in range(len(corpus)):
        if corpus[i] in burst_reviews:
            Y[i] = 1
    
    # LDA
    #n_components = nombre de topics
    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(X.toarray(), Y)
    
    print("Document 0: ")
    for i,topic in enumerate(X_lda):
      print("Topic ",i,": ",topic*100,"%")
      
############################# MAIN INSTRUCTIONS ##############################

# --- ETAPE 1: Parsing du fichier json
# data, fields = parse('data/data.json', 65000)
# data = doublonsNonSpams(data, fields)

# --- ETAPE 2: Algorithme PageRank et spammicité
# rg = ReviewGraph(data, fields, window = 1)
# rg.computeScores(niter=50)
# worst = rg.get_k_worst(50)
# best = rg.get_k_best(50)

# --- ETAPE 3: TAL
# tp = TextProcessor(corpus)
# tp.process(lower=True, remove_punc=True, remove_digits=True, normalize=True, remove_stopwords=True, stemming=True)
# tp.vectorize(vtype='tf-idf', analyzer='word', ngram_range=(1,1))
# sim_matrix = tp.similarity_matrix()
# sim_threshold = tp.similarity_threshold(threshold=0)
# sum_sim_threshold = tp.overall_threshold(threshold=6)

## Pour récupérer un corpus de revues:
# corpus = create_corpus(data, fields, id_prods, month_min, month_max)