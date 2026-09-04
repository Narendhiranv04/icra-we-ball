(define (domain vilain-workshop)
  (:requirements :strips :typing :negative-preconditions)
  (:types
    movable location - object
    driver fastener - movable
    storage surface target - location
  )
  (:predicates
    (at ?object - movable ?location - location)
    (holding ?object - movable)
    (handempty)
    (accessible ?location - location)
    (open ?storage - storage)
    (driver-compatible ?driver - driver ?fastener - fastener)
    (fits ?fastener - fastener ?target - target)
    (can-reach ?driver - driver ?target - target)
    (inserted ?fastener - fastener ?target - target)
    (fastened ?fastener - fastener ?target - target)
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

  (:action insert
    :parameters (?fastener - fastener ?target - target)
    :precondition (and (holding ?fastener) (fits ?fastener ?target))
    :effect (and
      (handempty)
      (inserted ?fastener ?target)
      (at ?fastener ?target)
      (not (holding ?fastener))
    )
  )

  (:action drive
    :parameters (?driver - driver ?fastener - fastener ?target - target)
    :precondition (and
      (holding ?driver)
      (inserted ?fastener ?target)
      (driver-compatible ?driver ?fastener)
      (fits ?fastener ?target)
      (can-reach ?driver ?target)
    )
    :effect (fastened ?fastener ?target)
  )

  (:action place-on
    :parameters (?object - movable ?surface - surface)
    :precondition (and (holding ?object) (accessible ?surface))
    :effect (and (handempty) (at ?object ?surface) (not (holding ?object)))
  )
)
