from posthog import Posthog
import os

posthog = Posthog('phc_YpKoFD7smPe4SXRtVyMW766uP9AjUwnuRJ8hh2EJcVv',
                  host='https://eu.posthog.com')


def capture_event(event_name, event_properties):
    if not os.environ.get('ANONYMIZED_TELEMETRY') == "False":
        posthog.capture("pluto_user", event_name, event_properties)