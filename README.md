# Predicting National Team Football Match Outcomes

[![PENGWIN2024TASK1](https://img.shields.io/badge/Deep%20Learning%3A%20Architectures%20%26%20Methods-FOOTBALL%20EM%20PREDICTIONS-blue)](https://pengwin.grand-challenge.org/)

This repository contains our code for our Deep Learning: Architectures \& Methods project. 

## TODOs
- [x] Script to train the base model on league football matches: [training_step_1.py](training_step_1.py)
- [ ] Weighting players by league: [training_step_2.py](training_step_2.py)
- [ ] Script to predict EM matches (i.e. outcomes for every group match, then KO-system until finals)


### TODOs for Daniel
Das hier dreht sich alles um Training 2:  
Das Modell ist fertig implemeniert, das Script in [training_step_2.py](training_step_2.py) dürfte funktionieren sobald du dich um die Daten gekümmert hast: Was fehlt ist:
- **Trainingsdaten:** Welchen Datensatz benutzen wir? Der braucht die "leagues" column; nationals_league.csv hat diese column beispielsweise nicht?!?
  - Ändere die Datensätze in Rows 21-22 [training_step_2.py](training_step_2.py)
- **Validation:**
  - Das ist aufwändiger! Wir wollen für die Evaluation in training_step_2.py so evaluieren, wie wir auch die EM Daten eingeben werden. D.h. wir geben als Input die Average performance der Spieler des letzten Jahres ein. Dasfür ist bis jetzt noch **nichts** implementiert. Offene TODOs die mir auf die Schnelle einfallen sind:
    - Welcher Datensatz? Der Datensatz sollte wahrscheinlich aus diesem Jahr stammen, damit wir einfach die durchschnittliche Performance dieses Jahres nehmen können?
    - Wie bekommen wir für jedes Spiel die Average performance der teilnehmenden Spieler als Input? Eventuell kannst du dir noch eine Dataset-Klasse ähnlich wie die in [dataset.py](dataset.py) bauen.
- **Finale Evaluation:** Nachdem [training_step_2.py](training_step_2.py) ausgeführt wurde, haben wir ein fertig trainiertes Modell. Das wollen wir auf zuvor noch nie gesehenen Daten evaluieren. Du solltest ein neues Script hierfür schreiben, kannst aber wahrscheinlich sehr viel deiner Arbeit aus der Validation wiederverwenden. Wir haben also einen Train/Validation/Test split:
  - Trainings-daten: Sind die Daten, auf denen wir trainieren, d.h. irgendwann overfitten wir.
  - Validation: Damit verhindern wir das Overfitting des Modells. Wenn wir aber mehrere Modelle in verschiedenen Konfigurationen trainieren, dann overfitten wir indirekt auf den Validation-Daten. Deshalb:
  - Test: Die finale Evaluation des Modells führen wir auf zuvor noch nie gesehenen Daten durch. Bspw. die aller-letzten Nation-league Spiele oder die EM 2020


## Authors
- Paul Andre Seitz
- Lars Damian Teubner
- Leon Arne Engländer
- Daniel Kirn