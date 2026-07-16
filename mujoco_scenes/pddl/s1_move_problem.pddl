(define (problem s1-mobile-move)
  (:domain kitchen-mobile-actions)

  (:objects
    fetch - robot
    home cupboard1 cupboard2 box - location
    table serving_table - region
    kettle coffee_jar sugar_jar spoon - object
    B1 - container
  )

  (:init
    (at fetch home)
    (base-motion-ready fetch)
    (hand-empty fetch)
    (object-at kettle home)
    (object-at coffee_jar home)
    (object-at sugar_jar home)
    (object-at spoon home)
    (graspable kettle)
    (graspable coffee_jar)
    (graspable sugar_jar)
    (graspable spoon)
    (container-at B1 box)
    (container-closed B1)
    (handle-graspable B1)

    ;; cupboard2 and box are symbolic aliases for the same right-side pose.
    (connected home cupboard1)
    (connected cupboard1 home)
    (connected home cupboard2)
    (connected cupboard2 home)
    (connected home box)
    (connected box home)
    (connected cupboard2 box)
    (connected box cupboard2)
  )

  (:goal (at fetch home))
)
