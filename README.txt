Requisiti:
- Ambiente con Python 3.7 o superiore.
- Scikit-learn, pandas, NumPy, Matplotlib installate.

Per eseguire la valutazione del nostro modello:
- Copiare il test set nella cartella in cui si trova questo README.
- Eseguire lo script "evaluation.py" avviando l'interprete Python nella suddetta cartella
  (IMPORTANTE, altrimenti fallirà il caricamento dei file necessari e l'importazione di
  funzioni e oggetti ausiliari dalla libreria "aux_lib.py"), in alternativa importare
  il file come modulo Python ed eseguire la funzione "evaluation()" al suo interno,
  avviando lo script principale sempre nella cartella specificata prima.
- La routine caricherà il nostro modello dal file "best_pipeline.sav"; quando richiesto,
  inserire il nome del file contenente il test set completo di estensione.
- La routine eseguirà lo split del test set in features e labels, chiamerà il metodo "predict"
  e stamperà l'F1 macro score e la matrice di confusione.
- Per terminare la routine: chiudere la finestra della matrice di confusione.

Il file "aux_lib.py" contiene la definizione di funzioni che calcolano e mostrano l'F1 macro score
e la matrice di confusione, e la definizione di una classe necessaria per il funzionamento del
modello caricato dal file "best_pipeline.sav".
Per i dettagli di tutti gli altri file presenti si rimanda al paragrafo "Contenuto della repository"
della relazione PDF.

Alessandro Tenaglia, Roberto Masocco
