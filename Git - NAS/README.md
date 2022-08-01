<h1>POC : Architecture Training</h1>

Pour associer la conception d'une architecture de modèle de Deep Learning et son entrainement dans la voie de la frugalité, nous avons choisi de nous orienter vers la recherche d'architecture automatique (NAS) s'incluant dans le contexte de l'AutoDL.

</br>
<h1>Recherche d'architecture</h1>

Le NAS (Neural Architecure Search) est un concept qui permet à travers différentes techniques de générer le modèle de deep learning optimal pour une tâche particulière. Au cours de cette période de POC, nous sommes parvenus à maîtriser 2 techniques de recherche d'architecture : une technique simple de NAS et une technique plus complexe appelée DARTS (Differentiable Architecture Search).

</br>
<h1>Organisation</h1>

Notre repository associé à ce POC est divisé en 2 parties :
- La partie [NAS](https://gitlab.aubay.io/development/innov/fla/poc/2022-s1/nas/-/tree/main/nas) qui contient notre travail effecuté pendant la première étape de cette phase de POC. Ce dossier contient une implémentation simple d'une recherche d'architecture.
- La partie [DARTS](https://gitlab.aubay.io/development/innov/fla/poc/2022-s1/nas/-/tree/main/darts) qui présente la solution plus performante pour la recherche d'architecture sur laquelle nous avons travaillé pendant la deuxième partie de cette phase de POC. 