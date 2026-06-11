import time
from actuation_policy import DrowsinessActivationPolicy

def run_drowsiness_test():
    print("--- Avvio Test: Drowsiness Activation Policy ---\n")
    
    # 1. Definiamo gli ID degli attuatori fittizi (simili a quelli che ti aspetteresti dal Manager)
    actuators_disponibili = ["meta_device_01", "speaker_bluetooth_01"]
    
    # 2. Inizializziamo la policy
    # Nota: la policy usa get_policy_attempts() da utils.config. 
    # Assicurati che il tuo file config.yaml sia configurato o restituirà il default.
    policy = DrowsinessActivationPolicy(actuator_ids=actuators_disponibili)
    
    # Forziamo a 2 i tentativi massimi di vibrazione per testare velocemente il passaggio allo speaker
    policy.max_vibration_attempts = 2 
    
    # 3. Creiamo uno scenario di eventi:
    # L'utente è sveglio -> inizia ad addormentarsi -> confermato più volte -> si sveglia
    scenario = [
        {"desc": "Utente Sveglio", "drowsiness_tag": 0},
        {"desc": "Sospetto sonno", "drowsiness_tag": 1},
        {"desc": "Sonno confermato (Tentativo 1)", "drowsiness_tag": 2},
        {"desc": "Sonno confermato (Tentativo 2)", "drowsiness_tag": 2},
        {"desc": "Sonno confermato (Passaggio Audio)", "drowsiness_tag": 2},
        {"desc": "Sonno confermato (Cooldown audio attivo)", "drowsiness_tag": 2},
        {"desc": "Utente di nuovo Sveglio", "drowsiness_tag": 0}
    ]
    
    # 4. Esecuzione del test
    for evento in scenario:
        print(f"\n[EVENTO IN INGRESSO]: {evento['desc']} (Tag: {evento['drowsiness_tag']})")
        
        # Passiamo l'evento alla policy
        azione = policy.handle(evento)
        
        if azione is None:
            print(" -> [RISPOSTA]: Nessuna attuazione (oppure cooldown attivo).")
        else:
            attuatore = azione['actuator_id']
            parametri = azione['params']
            
            if attuatore.startswith("meta_"):
                print(f" -> [AZIONE HAPTIC]: Attivazione Vibrazione su {attuatore}")
                print(f"    Parametri: Duty={parametri.get('duty')}, Duration={parametri.get('duration')}ms")
            elif attuatore.startswith("speaker_"):
                print(f" -> [AZIONE AUDIO]: Riproduzione Suono su {attuatore}")
                print(f"    File: {parametri.get('file')}")
                
        # Pausa per simulare il tempo che passa tra i frame/eventi della telecamera
        # Mettiamo 1 secondo per il test (il cooldown dello speaker nella tua policy è 5 secondi)
        time.sleep(1)

    print("\n--- Test completato ---")

if __name__ == "__main__":
    run_drowsiness_test()
