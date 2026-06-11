import time
from event_dispatcher import EventDispatcher
from actuator_manager import ActuatorManager
from utils.logger import log_system

def run_test():
    """
    Script di test per verificare il workflow:
    Evento -> EventDispatcher -> Policy -> ActuatorManager -> Attuatore
    """
    print("--- Inizializzazione Ambiente di Test ---")
    
    # 1. Inizializziamo il gestore degli attuatori
    manager = ActuatorManager()
    
    # 2. Inizializziamo il dispatcher
    # Supponendo che il dispatcher sia in grado di inviare eventi al manager
    dispatcher = EventDispatcher()
    
    # 3. Definiamo una lista di eventi di test da simulare
    test_events = [
        {"tag": "movement_detected", "priority": 1},
        {"tag": "system_error", "priority": 0}
    ]
    
    print(f"--- Lancio di {len(test_events)} eventi di prova ---")
    
    for event in test_events:
        print(f"\n[TEST] Invio evento: {event['tag']}")
        
        # Inseriamo l'evento nel sistema
        # In base alla logica del tuo progetto, il dispatcher dovrebbe 
        # notificare il manager o inserire l'evento in una coda
        dispatcher.dispatch(event)
        
        # Attendiamo che l'attuatore elabori l'evento
        time.sleep(3) 

    print("\n--- Test completato. Controlla i log per i dettagli esecutivi. ---")

if __name__ == "__main__":
    run_test()
