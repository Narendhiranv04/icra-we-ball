(define (domain vilain-living-room)
  (:requirements :strips :typing)
  (:types
    movable location seat - object
    cup saucer remote - movable
    support - location
  )
  (:predicates
    (at ?object - movable ?location - location)
    (holding ?object - movable)
    (handempty)
    (present ?location - location)
    (accessible ?location - location)
    (supports ?support - support ?object - movable)
    (personal-to ?support - support ?seat - seat)
    (shared ?support - support)
  )

  (:action pick-from
    :parameters (?object - movable ?location - location)
    :precondition (and (handempty) (present ?location) (accessible ?location) (at ?object ?location))
    :effect (and (holding ?object) (not (handempty)) (not (at ?object ?location)))
  )

  (:action place-on
    :parameters (?object - movable ?support - support)
    :precondition (and (holding ?object) (present ?support) (accessible ?support))
    :effect (and
      (handempty)
      (at ?object ?support)
      (supports ?support ?object)
      (not (holding ?object))
    )
  )
)
