# ml_project

PARTE A.

• Progettazione ed implementazione di funzioni per simulare la propagazione in avanti di unarete neurale multi-strato con almeno:   
  due strati di pesi, con  la  sigmoide come funzione dioutput dei nodi interni e l'identità come funzione di output dei nodi di output.
    (FACOLTATIVO: permettere all'utente di implementare reti con più di uno strato dinodi interni e con qualsiasi 
                  funzione di output per ciascun strato)

• Progettazione ed implementazione di funzioni per la realizzazione della back-propagationper reti neurali 
  multi-strato con almeno: due strati di pesi, con la sigmoide come funzione dioutput dei nodi interni  e l'identità come funzione
  di output dei nodi di output, con la sommadei quadrati come funzione di errore.
    (FACOLTATIVO: permettere all'utente di realizzare la back-propagation con più di uno strato di nodi interni, con qualsiasi 
                  funzione di output per ciascun strato e con  qualsiasi funzione di errore derivabile rispetto all'output).

PARTE B.

  Si consideri come input le immagine raw del dataset mnist. Si ha, allora,un problema di classificazione a C classi, con C=10. 
  Si estragga opportunamente un datasetglobale di N coppie (ad esempio, N=200).Si fissi la discesa del gradiente come algoritmodi
  aggiornamento dei pesi, ed una rete neurale con un unico strato di nodi interni. Siscelgano gli iper-parametri del modello, 
  cioè eta della regola di aggiornamento ed il numerodi nodi interni, sulla base di un approccio di cross-validation k-fold (ad esempio k=5).
  Scegliere e mantenere invariati tutti gli altri "parametri" come, ad esempio, le funzioni dioutput e la funzione di errore.
  Se è necessario, per questioni di tempi computazionali espazio in memoria, si possono ridurre (ad esempio dimezzarle) 
  le dimensioni delle immaginiraw del dataset mnist (ad esempio utilizzando in matlab la funzione imresize)
