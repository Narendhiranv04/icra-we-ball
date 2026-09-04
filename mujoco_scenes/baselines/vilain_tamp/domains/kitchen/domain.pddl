(define (domain vilain-kitchen)
  (:requirements :strips :typing :negative-preconditions)
  (:types
    entity content location - object
    movable - entity
    vessel source utensil - movable
    surface storage - location
  )
  (:predicates
    (at ?object - movable ?location - location)
    (holding ?object - movable)
    (handempty)
    (accessible ?location - location)
    (open ?storage - storage)
    (contains ?vessel - vessel ?content - content)
    (can-dispense ?source - source ?content - content)
    (can-stir ?utensil - utensil ?vessel - vessel)
    (can-serve-with ?utensil - utensil ?vessel - vessel)
    (inside ?utensil - utensil ?vessel - vessel)
    (stirred ?vessel - vessel)
  )

  (:action open-storage
    :parameters (?storage - storage)
    :precondition (not (open ?storage))
    :effect (and (open ?storage) (accessible ?storage))
  )

  (:action pick-from
    :parameters (?object - movable ?location - location)
    :precondition (and (handempty) (accessible ?location) (at ?object ?location))
    :effect (and (holding ?object) (not (handempty)) (not (at ?object ?location)))
  )

  (:action place-on
    :parameters (?object - movable ?surface - surface)
    :precondition (and (holding ?object) (accessible ?surface))
    :effect (and (handempty) (at ?object ?surface) (not (holding ?object)))
  )

  (:action pour
    :parameters (?source - source ?target - vessel ?content - content)
    :precondition (and (holding ?source) (can-dispense ?source ?content))
    :effect (contains ?target ?content)
  )

  (:action stir
    :parameters (?utensil - utensil ?target - vessel)
    :precondition (and (holding ?utensil) (can-stir ?utensil ?target))
    :effect (stirred ?target)
  )

  (:action place-in
    :parameters (?utensil - utensil ?target - vessel)
    :precondition (and (holding ?utensil) (can-serve-with ?utensil ?target))
    :effect (and (handempty) (inside ?utensil ?target) (not (holding ?utensil)))
  )
)
