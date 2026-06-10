import time

from actuators.actuator_manager import ActuatorManager

from core.actuation_policy import StereotipyActivationPolicy
from core.event_dispatcher import EventDispatcher

from utils.logger import log_system
from utils.config import get_bluecoin_config

def main():
    actuator_manager = None

    try:
        log_system("[MAIN] Initializing STOPme system...")

        actuator_manager = ActuatorManager()

        actuator_manager.scan_actuators()

        actuator_manager.initialize_actuators()

        actuators_list = actuator_manager.get_actuators_ids()

        if not actuators_list:
            log_system("[MAIN] No actuators discovered. Event detection and logging still executing")

        policy = StereotipyActivationPolicy(actuator_ids=actuators_list)

        dispatcher = EventDispatcher(
            actuator_manager=actuator_manager,
            policy=policy,
            yolo_thread=yolo_thread,
            movenet_thread=movenet_thread,
            roi_state=roi_state,
        )

        
    except KeyboardInterrupt:
        log_system("[TEST] Termination signal received.")

    except Exception as e:
        log_system(f"[TEST] Unhandled error: {e}", level="ERROR")

    finally:
        log_system("[TEST] Shutting down test...")
        
        if actuator_manager:
            actuator_manager.stop_all()

        if dashboard:
            unregister_dashboard_console()
            dashboard.close()
            
        log_system("[TEST] Shutdown complete.")

if __name__ == "__main__":
    main()
