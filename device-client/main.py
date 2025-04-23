import os
import argparse
import logging
import json
import time
from pathlib import Path

from src.device_client import DeviceClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("echonest_device.main")

def main():
    """
    Main entry point for the EchoNest AI device client.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="EchoNest AI Device Client")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--register", help="Register device with registration code")
    parser.add_argument("--device-name", help="Device name for registration")
    parser.add_argument("--sync", action="store_true", help="Trigger immediate synchronization")
    parser.add_argument("--force-sync", action="store_true", help="Force full synchronization")
    parser.add_argument("--chat", help="Process chat query")
    parser.add_argument("--session", help="Chat session ID")
    parser.add_argument("--language", help="Language code")
    parser.add_argument("--voice", help="Process voice query (path to audio file)")
    parser.add_argument("--info", action="store_true", help="Show device information")
    parser.add_argument("--sessions", action="store_true", help="List chat sessions")
    parser.add_argument("--messages", help="Show messages from chat session")
    parser.add_argument("--limit", type=int, help="Limit number of messages")
    
    args = parser.parse_args()
    
    try:
        # Initialize device client
        client = DeviceClient(config_path=args.config)
        
        # Register device if requested
        if args.register:
            result = client.register_device(
                registration_code=args.register,
                device_name=args.device_name
            )
            print(json.dumps(result, indent=2))
            return
        
        # Start client
        client.start()
        
        # Process commands
        if args.sync or args.force_sync:
            result = client.sync_now(force=args.force_sync)
            print(f"Sync queued: {result}")
        
        elif args.chat:
            result = client.process_chat(
                query=args.chat,
                session_id=args.session,
                language=args.language
            )
            print(json.dumps(result, indent=2))
        
        elif args.voice:
            result = client.process_voice(
                audio_path=args.voice,
                session_id=args.session,
                language=args.language
            )
            print(json.dumps(result, indent=2))
        
        elif args.info:
            info = client.get_device_info()
            print(json.dumps(info, indent=2))
        
        elif args.sessions:
            sessions = client.get_chat_sessions()
            print(json.dumps(sessions, indent=2))
        
        elif args.messages:
            messages = client.get_chat_messages(
                session_id=args.messages,
                limit=args.limit
            )
            print(json.dumps(messages, indent=2))
        
        else:
            # Run in interactive mode
            print("EchoNest AI Device Client")
            print("Type 'exit' to quit")
            
            while True:
                try:
                    command = input("> ")
                    
                    if command.lower() == "exit":
                        break
                    
                    if command.startswith("chat "):
                        query = command[5:]
                        result = client.process_chat(query=query)
                        print(f"Response: {result.get('response', {}).get('response')}")
                    
                    elif command == "info":
                        info = client.get_device_info()
                        print(json.dumps(info, indent=2))
                    
                    elif command == "sync":
                        result = client.sync_now()
                        print(f"Sync queued: {result}")
                    
                    elif command == "sessions":
                        sessions = client.get_chat_sessions()
                        print(json.dumps(sessions, indent=2))
                    
                    elif command.startswith("messages "):
                        session_id = command[9:]
                        messages = client.get_chat_messages(session_id=session_id)
                        print(json.dumps(messages, indent=2))
                    
                    else:
                        print("Unknown command")
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")
        
        # Stop client
        client.stop()
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
